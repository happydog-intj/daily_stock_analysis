# -*- coding: utf-8 -*-
"""
AkShare fundamental adapter (fail-open).

This adapter intentionally uses capability probing against multiple AkShare
endpoint candidates. It should never raise to caller; partial data is allowed.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_DIVIDEND_KEYWORD_MAP: Dict[str, List[str]] = {
    "per_share": [
        "每股派息",
        "每股现金红利",
        "每股分红",
        "每股派现",
        "派现(元/股)",
        "派息(元/股)",
        "税前派息(元/股)",
        "现金分红(税前)",
    ],
    "plan_text": [
        "分配方案",
        "分红方案",
        "实施方案",
        "派息方案",
        "方案",
        "预案",
        "方案说明",
    ],
    "ex_dividend_date": ["除权除息日", "除息日", "除权日", "除权除息", "除息日期"],
    "record_date": ["股权登记日", "登记日"],
    "announce_date": ["公告日期", "公告日", "实施公告日", "预案公告日"],
    "report_date": ["报告期", "报告日期", "截止日期", "统计截止日期"],
}


def _safe_float(value: Any) -> Optional[float]:
    """Best-effort float conversion."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    s = str(value).strip().replace(",", "").replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    try:
        return parsed.to_pydatetime()
    except Exception:
        return None


def _normalize_code(raw: Any) -> str:
    s = _safe_str(raw).upper()
    if "." in s:
        s = s.split(".", 1)[0]
    s = re.sub(r"^(SH|SZ|BJ)", "", s)
    return s


def _pick_by_keywords(row: pd.Series, keywords: List[str]) -> Optional[Any]:
    """
    Return first non-empty row value whose column name contains any keyword.
    """
    for col in row.index:
        col_s = str(col)
        if any(k in col_s for k in keywords):
            val = row.get(col)
            if val is not None and str(val).strip() not in ("", "-", "nan", "None"):
                return val
    return None


def _parse_dividend_plan_to_per_share(plan_text: str) -> Optional[float]:
    """Parse per-share cash dividend from Chinese plan text."""
    text = _safe_str(plan_text)
    if not text:
        return None

    for pattern in (
        r"(?:每)?\s*10\s*股?\s*派(?:发)?\s*([0-9]+(?:\.[0-9]+)?)\s*元",
        r"10\s*派\s*([0-9]+(?:\.[0-9]+)?)\s*元",
    ):
        match = re.search(pattern, text)
        if match:
            parsed = _safe_float(match.group(1))
            if parsed is not None and parsed > 0:
                return parsed / 10.0

    match_per_share = re.search(r"每\s*股\s*派(?:发)?\s*([0-9]+(?:\.[0-9]+)?)\s*元", text)
    if match_per_share:
        parsed = _safe_float(match_per_share.group(1))
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _extract_cash_dividend_per_share(row: pd.Series) -> Optional[float]:
    """Extract pre-tax cash dividend per share from a row."""
    plan_text = _safe_str(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["plan_text"]))
    # Keep pre-tax semantics; skip explicit after-tax plans unless pre-tax marker exists.
    if "税后" in plan_text and "税前" not in plan_text and "含税" not in plan_text:
        return None

    direct = _safe_float(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["per_share"]))
    if direct is not None and direct > 0:
        return direct
    return _parse_dividend_plan_to_per_share(plan_text)


def _filter_rows_by_code(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码", "symbol", "ts_code"))]
    if not code_cols:
        return df

    target = _normalize_code(stock_code)
    for col in code_cols:
        try:
            series = df[col].astype(str).map(_normalize_code)
            filtered = df[series == target]
            if not filtered.empty:
                return filtered
        except Exception:
            continue
    return pd.DataFrame()


def _normalize_report_date(value: Any) -> Optional[str]:
    parsed = _safe_datetime(value)
    return parsed.date().isoformat() if parsed else None


def _build_dividend_payload(
    dividend_df: pd.DataFrame,
    stock_code: str,
    max_events: int = 5,
) -> Dict[str, Any]:
    work_df = _filter_rows_by_code(dividend_df, stock_code)
    if work_df.empty:
        return {}

    now_date = datetime.now().date()
    ttm_start_date = now_date - timedelta(days=365)
    dedupe_keys = set()
    events: List[Dict[str, Any]] = []

    for _, row in work_df.iterrows():
        if not isinstance(row, pd.Series):
            continue
        ex_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["ex_dividend_date"]))
        record_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["record_date"]))
        announce_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["announce_date"]))
        event_dt = ex_dt or record_dt or announce_dt
        if event_dt is None:
            continue
        event_date = event_dt.date()
        if event_date > now_date:
            continue

        per_share = _extract_cash_dividend_per_share(row)
        if per_share is None or per_share <= 0:
            continue

        dedupe_key = (event_date.isoformat(), round(per_share, 6))
        if dedupe_key in dedupe_keys:
            continue
        dedupe_keys.add(dedupe_key)

        events.append(
            {
                "event_date": event_date.isoformat(),
                "ex_dividend_date": ex_dt.date().isoformat() if ex_dt else None,
                "record_date": record_dt.date().isoformat() if record_dt else None,
                "announcement_date": announce_dt.date().isoformat() if announce_dt else None,
                "cash_dividend_per_share": round(per_share, 6),
                "is_pre_tax": True,
            }
        )

    if not events:
        return {}

    events.sort(key=lambda item: item.get("event_date") or "", reverse=True)
    ttm_events: List[Dict[str, Any]] = []
    for item in events:
        event_dt = _safe_datetime(item.get("event_date"))
        if event_dt is None:
            continue
        event_date = event_dt.date()
        if ttm_start_date <= event_date <= now_date:
            ttm_events.append(item)

    return {
        "events": events[:max(1, max_events)],
        "ttm_event_count": len(ttm_events),
        "ttm_cash_dividend_per_share": (
            round(sum(float(item.get("cash_dividend_per_share") or 0.0) for item in ttm_events), 6)
            if ttm_events else None
        ),
        "coverage": "cash_dividend_pre_tax",
        "as_of": now_date.isoformat(),
    }


def _extract_latest_row(df: pd.DataFrame, stock_code: str) -> Optional[pd.Series]:
    """
    Select the most relevant row for the given stock.
    """
    if df is None or df.empty:
        return None

    code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码", "ts_code", "symbol"))]
    target = _normalize_code(stock_code)
    if code_cols:
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                matched = df[series == target]
                if not matched.empty:
                    return matched.iloc[0]
            except Exception:
                continue
        return None

    # Fallback: use latest row
    return df.iloc[0]


class AkshareFundamentalAdapter:
    """AkShare adapter for fundamentals, capital flow and dragon-tiger signals."""

    def _call_df_candidates(
        self,
        candidates: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], List[str]]:
        errors: List[str] = []
        try:
            import akshare as ak
        except Exception as exc:
            return None, None, [f"import_akshare:{type(exc).__name__}"]

        for func_name, kwargs in candidates:
            fn = getattr(ak, func_name, None)
            if fn is None:
                continue
            try:
                df = fn(**kwargs)
                if isinstance(df, pd.Series):
                    df = df.to_frame().T
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df, func_name, errors
            except Exception as exc:
                errors.append(f"{func_name}:{type(exc).__name__}")
                continue
        return None, None, errors

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        """
        Return normalized fundamental blocks from AkShare with partial tolerance.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "source_chain": [],
            "errors": [],
        }

        # Financial indicators
        # stock_financial_abstract returns a wide-format DataFrame:
        #   rows = financial indicators (identified by '指标' column)
        #   columns = reporting periods (e.g. '20260331', '20251231', ...)
        # We must look up values by indicator row name, NOT by column keyword.
        # stock_financial_analysis_indicator (fallback) returns a normal row-per-stock table.
        fin_df, fin_source, fin_errors = self._call_df_candidates([
            ("stock_financial_abstract", {"symbol": stock_code}),
            ("stock_financial_analysis_indicator", {"symbol": stock_code}),
            ("stock_financial_analysis_indicator", {}),
        ])
        result["errors"].extend(fin_errors)
        if fin_df is not None:
            # Detect wide-format: has '指标' column and date-like columns (8-digit strings)
            is_wide_format = (
                "指标" in fin_df.columns
                and any(
                    str(c).strip().isdigit() and len(str(c).strip()) == 8
                    for c in fin_df.columns
                )
            )
            if is_wide_format:
                # Wide format from stock_financial_abstract:
                # index by '指标', use the latest date column (first date column)
                date_cols = [
                    c for c in fin_df.columns
                    if str(c).strip().isdigit() and len(str(c).strip()) == 8
                ]
                if date_cols:
                    latest_col = date_cols[0]  # most recent period first
                    try:
                        df_idx = fin_df.drop_duplicates(subset=["指标"]).set_index("指标")
                    except Exception:
                        df_idx = fin_df.set_index("指标")

                    def _get_row_val(indicator: str) -> Optional[float]:
                        if indicator not in df_idx.index:
                            return None
                        return _safe_float(df_idx.loc[indicator, latest_col])

                    # YoY fields
                    revenue_yoy = _get_row_val("营业总收入增长率")
                    profit_yoy = _get_row_val("归属母公司净利润增长率")
                    roe = _get_row_val("净资产收益率(ROE)")
                    # gross_margin: prefer direct '毛利率' row; fallback to compute from revenue/cost
                    gross_margin: Optional[float] = _get_row_val("毛利率")
                    if gross_margin is None:
                        revenue_raw = _get_row_val("营业总收入")
                        op_cost_raw = _get_row_val("营业成本")
                        if revenue_raw is not None and op_cost_raw is not None and revenue_raw != 0:
                            gross_margin = round(
                                (revenue_raw - op_cost_raw) / revenue_raw * 100, 4
                            )
                    revenue_raw = _get_row_val("营业总收入")
                    net_profit_parent = _get_row_val("归母净利润")
                    operating_cash_flow = _get_row_val("经营现金流量净额")
                    # report_date: parse latest_col (YYYYMMDD)
                    rd_str = str(latest_col).strip()
                    if len(rd_str) == 8 and rd_str.isdigit():
                        report_date: Optional[str] = f"{rd_str[:4]}-{rd_str[4:6]}-{rd_str[6:8]}"
                    else:
                        report_date = _normalize_report_date(latest_col)
                    revenue = _safe_float(revenue_raw)
                else:
                    # Wide format but no date columns — fall back to row extraction
                    row = _extract_latest_row(fin_df, stock_code)
                    revenue_yoy = profit_yoy = roe = gross_margin = None
                    revenue = net_profit_parent = operating_cash_flow = None
                    report_date = None
                    if row is not None:
                        revenue_yoy = _safe_float(_pick_by_keywords(row, ["营业总收入增长率", "营业收入同比", "营收同比"]))
                        profit_yoy = _safe_float(_pick_by_keywords(row, ["归属母公司净利润增长率", "净利润同比"]))
                        roe = _safe_float(_pick_by_keywords(row, ["净资产收益率(ROE)", "净资产收益率", "ROE"]))
                        gross_margin = _safe_float(_pick_by_keywords(row, ["毛利率"]))
                        report_date = _normalize_report_date(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["report_date"]))
                        revenue = _safe_float(_pick_by_keywords(row, ["营业总收入", "营业收入", "营收"]))
                        net_profit_parent = _safe_float(_pick_by_keywords(row, ["归母净利润", "净利润"]))
                        operating_cash_flow = _safe_float(_pick_by_keywords(row, ["经营现金流量净额", "经营现金流"]))
            else:
                # Normal row-per-stock format (e.g. stock_financial_analysis_indicator)
                row = _extract_latest_row(fin_df, stock_code)
                revenue_yoy = profit_yoy = roe = gross_margin = None
                revenue = net_profit_parent = operating_cash_flow = None
                report_date = None
                if row is not None:
                    revenue_yoy = _safe_float(_pick_by_keywords(row, ["营业收入同比", "营收同比", "收入同比", "同比增长"]))
                    profit_yoy = _safe_float(_pick_by_keywords(row, ["净利润同比", "净利同比", "归母净利润同比"]))
                    roe = _safe_float(_pick_by_keywords(row, ["净资产收益率", "ROE", "净资产收益"]))
                    gross_margin = _safe_float(_pick_by_keywords(row, ["毛利率"]))
                    report_date = _normalize_report_date(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["report_date"]))
                    revenue = _safe_float(_pick_by_keywords(row, ["营业总收入", "营业收入", "营收"]))
                    net_profit_parent = _safe_float(_pick_by_keywords(row, ["归母净利润", "母公司股东净利润", "净利润"]))
                    operating_cash_flow = _safe_float(
                        _pick_by_keywords(row, ["经营活动产生的现金流量净额", "经营现金流", "经营活动现金流"])
                    )

            result["growth"] = {
                "revenue_yoy": revenue_yoy,
                "net_profit_yoy": profit_yoy,
                "roe": roe,
                "gross_margin": gross_margin,
            }
            financial_report_payload = {
                "report_date": report_date,
                "revenue": round(revenue / 1e8, 2) if revenue is not None else None,
                "net_profit_parent": round(net_profit_parent / 1e8, 2) if net_profit_parent is not None else None,
                "operating_cash_flow": round(operating_cash_flow / 1e8, 2) if operating_cash_flow is not None else None,
                "roe": roe,
            }
            if any(v is not None for v in financial_report_payload.values()):
                result["earnings"]["financial_report"] = financial_report_payload
            result["source_chain"].append(f"growth:{fin_source}")

        # Earnings forecast
        forecast_df, forecast_source, forecast_errors = self._call_df_candidates([
            ("stock_yjyg_em", {"symbol": stock_code}),
            ("stock_yjyg_em", {}),
            ("stock_yjbb_em", {"symbol": stock_code}),
            ("stock_yjbb_em", {}),
        ])
        result["errors"].extend(forecast_errors)
        if forecast_df is not None:
            row = _extract_latest_row(forecast_df, stock_code)
            if row is not None:
                result["earnings"]["forecast_summary"] = _safe_str(
                    _pick_by_keywords(row, ["预告", "业绩变动", "内容", "摘要", "公告"])
                )[:200]
                result["source_chain"].append(f"earnings_forecast:{forecast_source}")

        # Earnings quick report
        quick_df, quick_source, quick_errors = self._call_df_candidates([
            ("stock_yjkb_em", {"symbol": stock_code}),
            ("stock_yjkb_em", {}),
        ])
        result["errors"].extend(quick_errors)
        if quick_df is not None:
            row = _extract_latest_row(quick_df, stock_code)
            if row is not None:
                result["earnings"]["quick_report_summary"] = _safe_str(
                    _pick_by_keywords(row, ["快报", "摘要", "公告", "说明"])
                )[:200]
                result["source_chain"].append(f"earnings_quick:{quick_source}")

        # Dividend details (cash dividend, pre-tax)
        dividend_df, dividend_source, dividend_errors = self._call_df_candidates([
            ("stock_fhps_detail_em", {"symbol": stock_code}),
            ("stock_history_dividend_detail", {"symbol": stock_code, "indicator": "分红", "date": ""}),
            ("stock_dividend_cninfo", {"symbol": stock_code}),
        ])
        result["errors"].extend(dividend_errors)
        if dividend_df is not None:
            dividend_payload = _build_dividend_payload(dividend_df, stock_code, max_events=5)
            if dividend_payload:
                result["earnings"]["dividend"] = dividend_payload
                result["source_chain"].append(f"dividend:{dividend_source}")

        # Institution / top shareholders
        inst_df, inst_source, inst_errors = self._call_df_candidates([
            ("stock_institute_hold", {}),
            ("stock_institute_recommend", {}),
        ])
        result["errors"].extend(inst_errors)
        if inst_df is not None:
            row = _extract_latest_row(inst_df, stock_code)
            if row is not None:
                inst_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "变动", "持股变化"]))
                result["institution"]["institution_holding_change"] = inst_change
                result["source_chain"].append(f"institution:{inst_source}")

        top10_df, top10_source, top10_errors = self._call_df_candidates([
            ("stock_gdfx_top_10_em", {"symbol": stock_code}),
            ("stock_gdfx_top_10_em", {}),
            ("stock_zh_a_gdhs_detail_em", {"symbol": stock_code}),
            ("stock_zh_a_gdhs_detail_em", {}),
        ])
        result["errors"].extend(top10_errors)
        if top10_df is not None:
            row = _extract_latest_row(top10_df, stock_code)
            if row is not None:
                holder_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "持股变化", "变动"]))
                result["institution"]["top10_holder_change"] = holder_change
                result["source_chain"].append(f"top10:{top10_source}")

        has_content = bool(result["growth"] or result["earnings"] or result["institution"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_three_statements(self, stock_code: str) -> Dict[str, Any]:
        """
        Fetch income statement, balance sheet and cash flow statement core metrics
        via akshare (stock_financial_report_sina).  A-share only; returns
        status='not_supported' for HK/US stocks.  Always fail-open.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "data": {},
        }
        try:
            import akshare as ak
        except Exception:
            result["status"] = "failed"
            return result

        # Only A-share 6-digit codes are supported by this source.
        code = _normalize_code(stock_code)
        if not re.match(r"^\d{6}$", code):
            return result  # HK / US – not_supported

        def _v(df: pd.DataFrame, col: str, row_idx: int = 0):
            """Safe value extractor from a df row."""
            if df is None or df.empty:
                return None
            if col not in df.columns:
                return None
            val = df[col].iloc[row_idx]
            return _safe_float(val)

        def _pct(numerator, denominator) -> Optional[float]:
            if numerator is None or denominator is None:
                return None
            try:
                d = float(denominator)
                if d == 0:
                    return None
                return round(float(numerator) / d * 100, 4)
            except (TypeError, ValueError):
                return None

        def _bn(val) -> Optional[float]:
            """Convert raw yuan to 亿元, rounded to 2 dp."""
            if val is None:
                return None
            try:
                return round(float(val) / 1e8, 2)
            except (TypeError, ValueError):
                return None

        try:
            # ── 利润表 ────────────────────────────────────────────────
            income_df = ak.stock_financial_report_sina(stock=code, symbol="利润表")
            income: Dict[str, Any] = {}
            if income_df is not None and not income_df.empty:
                revenue      = _v(income_df, "营业总收入")
                op_cost      = _v(income_df, "营业成本")
                op_profit    = _v(income_df, "营业利润")
                net_profit   = _v(income_df, "净利润")
                sell_exp     = _v(income_df, "销售费用")
                admin_exp    = _v(income_df, "管理费用")
                rd_exp       = _v(income_df, "研发费用")
                report_date_raw = income_df["报告日"].iloc[0] if "报告日" in income_df.columns else None
                # Normalise report date: e.g. 20251231 -> 2025-12-31
                report_date: Optional[str] = None
                if report_date_raw is not None:
                    s = str(report_date_raw).strip()
                    if re.match(r"^\d{8}$", s):
                        report_date = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
                    else:
                        report_date = _normalize_report_date(report_date_raw)

                gross_profit = (float(revenue) - float(op_cost)) if (revenue is not None and op_cost is not None) else None
                gross_margin = _pct(gross_profit, revenue)
                net_margin   = _pct(net_profit, revenue)
                sell_ratio   = _pct(sell_exp, revenue)
                admin_ratio  = _pct(admin_exp, revenue)
                rd_ratio     = _pct(rd_exp, revenue)

                income = {
                    "revenue":       _bn(revenue),
                    "op_cost":       _bn(op_cost),
                    "gross_margin":  gross_margin,
                    "op_profit":     _bn(op_profit),
                    "net_profit":    _bn(net_profit),
                    "sell_expense":  _bn(sell_exp),
                    "admin_expense": _bn(admin_exp),
                    "rd_expense":    _bn(rd_exp),
                    "sell_ratio":    sell_ratio,
                    "admin_ratio":   admin_ratio,
                    "rd_ratio":      rd_ratio,
                    "net_margin":    net_margin,
                    "report_date":   report_date,
                }

            # ── 资产负债表 ────────────────────────────────────────────
            balance_df = ak.stock_financial_report_sina(stock=code, symbol="资产负债表")
            balance: Dict[str, Any] = {}
            if balance_df is not None and not balance_df.empty:
                total_assets    = _v(balance_df, "资产总计")
                total_liab      = _v(balance_df, "负债合计")
                equity          = _v(balance_df, "所有者权益(或股东权益)合计")
                curr_assets     = _v(balance_df, "流动资产合计")
                curr_liab       = _v(balance_df, "流动负债合计")
                cash            = _v(balance_df, "货币资金")
                # Prefer combined field; fallback to standalone
                ar_val = _v(balance_df, "应收账款")
                if ar_val is None:
                    ar_val = _v(balance_df, "应收票据及应收账款")
                inventory       = _v(balance_df, "存货")

                debt_ratio      = _pct(total_liab, total_assets)
                current_ratio: Optional[float] = None
                if curr_assets is not None and curr_liab is not None:
                    try:
                        cl = float(curr_liab)
                        current_ratio = round(float(curr_assets) / cl, 4) if cl != 0 else None
                    except (TypeError, ValueError):
                        pass

                balance = {
                    "total_assets":  _bn(total_assets),
                    "total_liab":    _bn(total_liab),
                    "net_assets":    _bn(equity),
                    "debt_ratio":    debt_ratio,
                    "curr_assets":   _bn(curr_assets),
                    "curr_liab":     _bn(curr_liab),
                    "current_ratio": current_ratio,
                    "cash":          _bn(cash),
                    "accounts_recv": _bn(ar_val),
                    "inventory":     _bn(inventory),
                }

            # ── 现金流量表 ────────────────────────────────────────────
            cf_df = ak.stock_financial_report_sina(stock=code, symbol="现金流量表")
            cashflow: Dict[str, Any] = {}
            if cf_df is not None and not cf_df.empty:
                op_cf    = _v(cf_df, "经营活动产生的现金流量净额")
                inv_cf   = _v(cf_df, "投资活动产生的现金流量净额")
                fin_cf   = _v(cf_df, "筹资活动产生的现金流量净额")
                capex    = _v(cf_df, "购建固定资产、无形资产和其他长期资产所支付的现金")

                free_cf: Optional[float] = None
                if op_cf is not None and capex is not None:
                    try:
                        free_cf = round((float(op_cf) - float(capex)) / 1e8, 2)
                    except (TypeError, ValueError):
                        pass

                # Cash quality = operating CF / net profit
                net_profit_for_cf = income.get("net_profit")  # already in 亿元
                cash_quality: Optional[float] = None
                if op_cf is not None and net_profit_for_cf is not None:
                    try:
                        np_bn = float(net_profit_for_cf)
                        if np_bn != 0:
                            cash_quality = round(float(op_cf) / 1e8 / np_bn, 4)
                    except (TypeError, ValueError):
                        pass

                cashflow = {
                    "op_cf":        _bn(op_cf),
                    "inv_cf":       _bn(inv_cf),
                    "fin_cf":       _bn(fin_cf),
                    "free_cf":      free_cf,
                    "cash_quality": cash_quality,
                }

            # Determine report_date: prefer income statement's, then balance sheet
            report_date_final = (
                income.get("report_date")
                or (
                    _normalize_report_date(balance_df["报告日"].iloc[0])
                    if balance_df is not None and not balance_df.empty and "报告日" in balance_df.columns
                    else None
                )
            )

            has_data = bool(income or balance or cashflow)
            result["status"] = "ok" if has_data else "failed"
            result["data"] = {
                "income":      income,
                "balance":     balance,
                "cashflow":    cashflow,
                "report_date": report_date_final,
            }

        except Exception as exc:
            logger.warning("get_three_statements(%s) failed: %s", stock_code, exc)
            result["status"] = "failed"

        return result

    def get_historical_financials(self, stock_code: str, n_periods: int = 4) -> Dict[str, Any]:
        """
        Return multi-period (recent 4 quarters or annual) core financial metrics for trend analysis.

        Only supported for A-share codes (港股/美股 not supported).

        Returns a dict with key 'historical_financials' containing:
          status: 'ok' | 'failed'
          periods: list of period strings (YYYY-MM-DD)
          revenue, net_profit, operating_cf: lists in 亿元
          revenue_yoy, net_profit_yoy: lists in %
          gross_margin, roe: lists in %
          data_type: 'quarterly' | 'annual'
        """
        result: Dict[str, Any] = {
            "status": "failed",
            "periods": [],
            "revenue": [],
            "revenue_yoy": [],
            "net_profit": [],
            "net_profit_yoy": [],
            "gross_margin": [],
            "roe": [],
            "operating_cf": [],
            "data_type": "quarterly",
        }

        # Only A-share codes
        code_upper = _safe_str(stock_code).upper()
        if any(code_upper.startswith(pfx) for pfx in ("HK", "US")) or code_upper.endswith(".HK"):
            return result

        try:
            import akshare as ak
        except Exception as exc:
            logger.warning("[historical_financials] akshare import failed: %s", exc)
            return result

        try:
            df = ak.stock_financial_abstract(symbol=stock_code)
        except Exception as exc:
            logger.warning("[historical_financials] stock_financial_abstract(%s) failed: %s", stock_code, exc)
            return result

        if df is None or df.empty:
            return result

        try:
            # Date columns are all columns except '选项' and '指标'
            date_cols = [c for c in df.columns if c not in ("选项", "指标")]
            if not date_cols:
                return result

            # Normalize date column names to YYYY-MM-DD
            def _norm_date_col(raw: str) -> Optional[str]:
                s = str(raw).strip()
                # Format: 20240930 -> 2024-09-30
                if len(s) == 8 and s.isdigit():
                    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
                # Already formatted
                try:
                    return pd.to_datetime(s).date().isoformat()
                except Exception:
                    return None

            # Use first occurrence of each indicator (handle duplicate rows)
            df_indexed = df.drop_duplicates(subset=["指标"]).set_index("指标")

            # We need n_periods + 1 cols to compute YoY (but abstract df already has YoY)
            # Use up to n_periods columns; skip NaT cols
            valid_date_cols: List[str] = []
            valid_norm_dates: List[str] = []
            for col in date_cols:
                nd = _norm_date_col(col)
                if nd is not None:
                    valid_date_cols.append(col)
                    valid_norm_dates.append(nd)
                if len(valid_date_cols) >= n_periods:
                    break

            if not valid_date_cols:
                return result

            def _get_indicator_values(indicator: str, cols: List[str]) -> List[Optional[float]]:
                if indicator not in df_indexed.index:
                    return [None] * len(cols)
                row = df_indexed.loc[indicator]
                vals: List[Optional[float]] = []
                for col in cols:
                    if col in row.index:
                        v = row[col]
                        f = _safe_float(v)
                        vals.append(f)
                    else:
                        vals.append(None)
                return vals

            # Gross margin from abstract: row '毛利率', value is already in %
            revenue_raw = _get_indicator_values("营业总收入", valid_date_cols)
            net_profit_raw = _get_indicator_values("归母净利润", valid_date_cols)
            operating_cf_raw = _get_indicator_values("经营现金流量净额", valid_date_cols)
            revenue_yoy_raw = _get_indicator_values("营业总收入增长率", valid_date_cols)
            net_profit_yoy_raw = _get_indicator_values("归属母公司净利润增长率", valid_date_cols)
            roe_raw = _get_indicator_values("净资产收益率(ROE)", valid_date_cols)
            gross_margin_raw = _get_indicator_values("毛利率", valid_date_cols)

            def _to_yi(v: Optional[float]) -> Optional[float]:
                """Convert yuan to 亿元 (÷1e8), round to 2 decimal places."""
                if v is None:
                    return None
                return round(v / 1e8, 2)

            def _round2(v: Optional[float]) -> Optional[float]:
                return round(v, 2) if v is not None else None

            result["periods"] = valid_norm_dates
            result["revenue"] = [_to_yi(v) for v in revenue_raw]
            result["net_profit"] = [_to_yi(v) for v in net_profit_raw]
            result["operating_cf"] = [_to_yi(v) for v in operating_cf_raw]
            result["revenue_yoy"] = [_round2(v) for v in revenue_yoy_raw]
            result["net_profit_yoy"] = [_round2(v) for v in net_profit_yoy_raw]
            result["roe"] = [_round2(v) for v in roe_raw]
            result["gross_margin"] = [_round2(v) for v in gross_margin_raw]

            # Determine data_type based on whether most periods end in -12-31 (annual)
            annual_count = sum(1 for d in valid_norm_dates if d.endswith("-12-31"))
            result["data_type"] = "annual" if annual_count > len(valid_norm_dates) / 2 else "quarterly"

            # Consider ok if at least periods and revenue are non-empty
            if result["periods"] and any(v is not None for v in result["revenue"]):
                result["status"] = "ok"

        except Exception as exc:
            logger.warning("[historical_financials] processing failed for %s: %s", stock_code, exc)
            result["status"] = "failed"

        return result

    def get_capital_flow(self, stock_code: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Return stock + sector capital flow.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "stock_flow": {},
            "sector_rankings": {"top": [], "bottom": []},
            "source_chain": [],
            "errors": [],
        }

        stock_df, stock_source, stock_errors = self._call_df_candidates([
            ("stock_individual_fund_flow", {"stock": stock_code}),
            ("stock_individual_fund_flow", {"symbol": stock_code}),
            ("stock_individual_fund_flow", {}),
            ("stock_main_fund_flow", {"symbol": stock_code}),
            ("stock_main_fund_flow", {}),
        ])
        result["errors"].extend(stock_errors)
        if stock_df is not None:
            row = _extract_latest_row(stock_df, stock_code)
            if row is not None:
                net_inflow = _safe_float(_pick_by_keywords(row, ["主力净流入", "净流入", "净额"]))
                inflow_5d = _safe_float(_pick_by_keywords(row, ["5日", "五日"]))
                inflow_10d = _safe_float(_pick_by_keywords(row, ["10日", "十日"]))
                result["stock_flow"] = {
                    "main_net_inflow": net_inflow,
                    "inflow_5d": inflow_5d,
                    "inflow_10d": inflow_10d,
                }
                result["source_chain"].append(f"capital_stock:{stock_source}")

        sector_df, sector_source, sector_errors = self._call_df_candidates([
            ("stock_sector_fund_flow_rank", {}),
            ("stock_sector_fund_flow_summary", {}),
        ])
        result["errors"].extend(sector_errors)
        if sector_df is not None:
            name_col = next((c for c in sector_df.columns if any(k in str(c) for k in ("板块", "行业", "名称", "name"))), None)
            flow_col = next((c for c in sector_df.columns if any(k in str(c) for k in ("净流入", "主力", "flow", "净额"))), None)
            if name_col and flow_col:
                work_df = sector_df[[name_col, flow_col]].copy()
                work_df[flow_col] = pd.to_numeric(work_df[flow_col], errors="coerce")
                work_df = work_df.dropna(subset=[flow_col])
                top_df = work_df.nlargest(top_n, flow_col)
                bottom_df = work_df.nsmallest(top_n, flow_col)
                result["sector_rankings"] = {
                    "top": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in top_df.iterrows()],
                    "bottom": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in bottom_df.iterrows()],
                }
                result["source_chain"].append(f"capital_sector:{sector_source}")

        has_content = bool(result["stock_flow"] or result["sector_rankings"]["top"] or result["sector_rankings"]["bottom"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_dragon_tiger_flag(self, stock_code: str, lookback_days: int = 20) -> Dict[str, Any]:
        """
        Return dragon-tiger signal in lookback window.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "is_on_list": False,
            "recent_count": 0,
            "latest_date": None,
            "source_chain": [],
            "errors": [],
        }

        df, source, errors = self._call_df_candidates([
            ("stock_lhb_stock_statistic_em", {}),
            ("stock_lhb_detail_em", {}),
            ("stock_lhb_jgmmtj_em", {}),
        ])
        result["errors"].extend(errors)
        if df is None:
            return result

        # Try code filter
        code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码"))]
        target = _normalize_code(stock_code)
        matched = pd.DataFrame()
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                cur = df[series == target]
                if not cur.empty:
                    matched = cur
                    break
            except Exception:
                continue
        if matched.empty:
            result["source_chain"].append(f"dragon_tiger:{source}")
            result["status"] = "ok" if code_cols else "partial"
            return result

        date_col = next((c for c in matched.columns if any(k in str(c) for k in ("日期", "上榜", "交易日", "time"))), None)
        parsed_dates: List[datetime] = []
        if date_col is not None:
            for val in matched[date_col].astype(str).tolist():
                try:
                    parsed_dates.append(pd.to_datetime(val).to_pydatetime())
                except Exception:
                    continue
        now = datetime.now()
        start = now - timedelta(days=max(1, lookback_days))
        recent_dates = [d for d in parsed_dates if start <= d <= now]

        result["is_on_list"] = bool(recent_dates)
        result["recent_count"] = len(recent_dates) if recent_dates else int(len(matched))
        result["latest_date"] = max(recent_dates).date().isoformat() if recent_dates else (
            max(parsed_dates).date().isoformat() if parsed_dates else None
        )
        result["status"] = "ok"
        result["source_chain"].append(f"dragon_tiger:{source}")
        return result

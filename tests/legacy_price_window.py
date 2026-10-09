# Unmodified function extracted from LangGraph.ipynb cell 47.
# Offline regression baseline only; not used by the repaired notebook.
def get_price_window(ticker: str, ref_date, window: int = 5):
    """
    ref_date 기준 [ref_date-window, ref_date] 구간의 수익률 요약.
    - price_df: ['date_kst','ticker','adj_close'] (date_kst는 tz-naive)
    - 범위 밖 ref_date는 min/max로 보정
    - 영업일 부족 시 fallback로 최근 포인트로 대체
    """
    global price_df
    if not ticker:
        return None

    # tz-naive로 통일
    if isinstance(ref_date, str):
        ref_date = pd.to_datetime(ref_date, utc=True, errors="coerce")
    if isinstance(ref_date, pd.Timestamp):
        if ref_date.tzinfo is not None:
            try:
                ref_date = ref_date.tz_convert(None)
            except Exception:
                ref_date = ref_date.tz_localize(None)

    sub = price_df[price_df["ticker"].str.upper() == ticker.upper()].copy()
    if sub.empty:
        return None

    sub = sub.sort_values("date_kst")
    dmin, dmax = sub["date_kst"].min(), sub["date_kst"].max()

    # 기준일 보정
    if ref_date < dmin:
        ref_date = dmin
    elif ref_date > dmax:
        ref_date = dmax

    start_date = ref_date - pd.Timedelta(days=window)

    # ✅ 초기 슬라이스 (반드시 먼저 생성)
    win = sub[(sub["date_kst"] >= start_date) & (sub["date_kst"] <= ref_date)].copy()

    # ✅ 영업일/공휴일로 부족하면 단계적 fallback
    if len(win) < 2:
        # ref_date 이전의 최근 포인트들 (window+1개 시도)
        fallback = sub[sub["date_kst"] <= ref_date].tail(max(2, window + 1))
        if len(fallback) >= 2:
            win = fallback
        else:
            # 그래도 부족하면 전체 마지막 포인트들
            fallback2 = sub.tail(max(2, window + 1))
            if len(fallback2) >= 2:
                win = fallback2
            else:
                return None

    win = win.sort_values("date_kst")
    start_p = float(win.iloc[0]["adj_close"])
    end_p   = float(win.iloc[-1]["adj_close"])
    pct     = (end_p / start_p - 1.0) * 100.0

    return {
        "ticker": ticker.upper(),
        "start_date": pd.to_datetime(win.iloc[0]["date_kst"]).date().isoformat(),
        "end_date": pd.to_datetime(win.iloc[-1]["date_kst"]).date().isoformat(),
        "start_price": round(start_p, 2),
        "end_price": round(end_p, 2),
        "pct_change": round(pct, 2),
        "n_points": int(len(win)),
    }

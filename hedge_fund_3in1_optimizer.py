import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

G_DF: Optional[pd.DataFrame] = None


@dataclass
class Signal:
    family: str
    direction: int
    score: float
    entry_price: float
    stop_price: float
    target_price: float
    reason: str


@dataclass
class Trade:
    family: str
    entry_idx: int
    exit_idx: int
    direction: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    result_points: float
    result_r: float
    exit_reason: str
    reason: str


@dataclass
class Metrics:
    trades: int
    wins: int
    losses: int
    timeouts: int
    win_rate: float
    total_points: float
    total_r: float
    avg_r: float
    expectancy_r: float
    profit_factor: float
    max_dd_r: float
    avg_hold_bars: float


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {c.lower().strip(): c for c in df.columns}
    rename_map = {}

    aliases = {
        "open": ["open", "abertura"],
        "high": ["high", "max", "maximum", "alta"],
        "low": ["low", "min", "minimum", "baixa"],
        "close": ["close", "fechamento"],
        "volume": ["volume", "tick_volume", "vol"],
        "time": ["time", "datetime", "date", "timestamp"],
    }

    for target, candidates in aliases.items():
        for c in candidates:
            if c in lower_map:
                rename_map[lower_map[c]] = target
                break

    df = df.rename(columns=rename_map)

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0

    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"])
        except Exception:
            pass

    return df.reset_index(drop=True)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV não encontrado: {path}")
    df = pd.read_csv(path)
    return normalize_columns(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["range"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()
    out["body_dir"] = np.sign(out["close"] - out["open"])

    prev_close = out["close"].shift(1)
    tr = np.maximum(
        out["high"] - out["low"],
        np.maximum((out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()),
    )
    out["tr"] = tr
    out["atr_14"] = out["tr"].rolling(14, min_periods=1).mean()
    out["atr_50"] = out["tr"].rolling(50, min_periods=1).mean()
    out["atr_ratio"] = (
        (out["atr_14"] / out["atr_50"].replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    for span in [5, 9, 13, 21, 34, 50, 100]:
        out[f"ema_{span}"] = out["close"].ewm(span=span, adjust=False).mean()

    out["ema_21_slope"] = out["ema_21"].diff().fillna(0.0)
    out["ema_50_slope"] = out["ema_50"].diff().fillna(0.0)

    for lb in [3, 5, 8, 13, 21, 34]:
        out[f"hh_{lb}"] = out["high"].rolling(lb, min_periods=1).max().shift(1)
        out[f"ll_{lb}"] = out["low"].rolling(lb, min_periods=1).min().shift(1)

    out["vol_ma_20"] = out["volume"].rolling(20, min_periods=1).mean()
    out["recent_high_10"] = out["high"].rolling(10, min_periods=1).max().shift(1)
    out["recent_low_10"] = out["low"].rolling(10, min_periods=1).min().shift(1)

    out["swing_high_3"] = (
        (out["high"] > out["high"].shift(1))
        & (out["high"] > out["high"].shift(-1))
    ).astype(int)

    out["swing_low_3"] = (
        (out["low"] < out["low"].shift(1))
        & (out["low"] < out["low"].shift(-1))
    ).astype(int)

    return out


def split_walkforward(n: int, train_pct: float = 0.60, valid_pct: float = 0.20) -> Dict[str, Tuple[int, int]]:
    train_end = int(n * train_pct)
    valid_end = int(n * (train_pct + valid_pct))
    return {
        "train": (0, train_end),
        "valid": (train_end, valid_end),
        "test": (valid_end, n),
    }


def in_session(ts, start_hour: int, end_hour: int) -> bool:
    if ts is None or pd.isna(ts):
        return True
    h = pd.Timestamp(ts).hour
    return start_hour <= h <= end_hour


def calc_breakout_signal(df: pd.DataFrame, i: int, p: Dict) -> Optional[Signal]:
    if i < p["bo_lookback"] + 2:
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    ts = row["time"] if "time" in df.columns else None

    if p["use_session"] and not in_session(ts, p["session_start"], p["session_end"]):
        return None

    if prev["range"] < p["bo_min_range"]:
        return None
    if prev["body"] < p["bo_min_body"]:
        return None
    if prev["atr_ratio"] < p["bo_min_atr_ratio"]:
        return None

    fast = prev[f"ema_{p['ema_fast']}"]
    slow = prev[f"ema_{p['ema_slow']}"]
    slope = prev["ema_21_slope"]

    hh = prev[f"hh_{p['bo_lookback']}"]
    ll = prev[f"ll_{p['bo_lookback']}"]

    score = 0.0

    if fast > slow:
        score += 1.5
    elif fast < slow:
        score += 1.5

    if abs(slope) >= p["bo_slope_min"]:
        score += 1.0

    if prev["body"] >= prev["atr_14"] * p["bo_expansion_body_atr"]:
        score += 2.0

    if prev["volume"] >= prev["vol_ma_20"] * p["bo_vol_mult"]:
        score += 0.5

    if row["high"] >= hh + p["bo_buffer"]:
        if p["bo_require_trend"] and not (fast > slow):
            return None
        entry = hh + p["bo_buffer"] + p["slippage"]
        stop = entry - p["bo_stop_points"]
        target = entry + p["bo_stop_points"] * p["bo_rr"]
        return Signal(
            family="breakout",
            direction=1,
            score=score,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            reason="breakout_long",
        )

    if row["low"] <= ll - p["bo_buffer"]:
        if p["bo_require_trend"] and not (fast < slow):
            return None
        entry = ll - p["bo_buffer"] - p["slippage"]
        stop = entry + p["bo_stop_points"]
        target = entry - p["bo_stop_points"] * p["bo_rr"]
        return Signal(
            family="breakout",
            direction=-1,
            score=score,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            reason="breakout_short",
        )

    return None


def calc_sweep_signal(df: pd.DataFrame, i: int, p: Dict) -> Optional[Signal]:
    if i < 12:
        return None

    row = df.iloc[i]
    ts = row["time"] if "time" in df.columns else None

    if p["use_session"] and not in_session(ts, p["session_start"], p["session_end"]):
        return None

    recent_high = row["recent_high_10"]
    recent_low = row["recent_low_10"]
    atr = row["atr_14"]

    if pd.isna(recent_high) or pd.isna(recent_low) or pd.isna(atr):
        return None

    score = 0.0

    if row["high"] > recent_high + p["sw_sweep_buffer"] and row["close"] < recent_high:
        score += 2.0
        if row["body"] >= atr * p["sw_rejection_body_atr"]:
            score += 1.5
        if row["close"] < row["open"]:
            score += 1.5

        entry = row["close"] - p["slippage"]
        stop = row["high"] + p["sw_stop_buffer"]
        risk = stop - entry
        if risk <= 0:
            return None
        target = entry - risk * p["sw_rr"]

        return Signal(
            family="sweep",
            direction=-1,
            score=score,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            reason="sweep_short",
        )

    if row["low"] < recent_low - p["sw_sweep_buffer"] and row["close"] > recent_low:
        score += 2.0
        if row["body"] >= atr * p["sw_rejection_body_atr"]:
            score += 1.5
        if row["close"] > row["open"]:
            score += 1.5

        entry = row["close"] + p["slippage"]
        stop = row["low"] - p["sw_stop_buffer"]
        risk = entry - stop
        if risk <= 0:
            return None
        target = entry + risk * p["sw_rr"]

        return Signal(
            family="sweep",
            direction=1,
            score=score,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            reason="sweep_long",
        )

    return None


def calc_smc_signal(df: pd.DataFrame, i: int, p: Dict) -> Optional[Signal]:
    if i < 25:
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    ts = row["time"] if "time" in df.columns else None

    if p["use_session"] and not in_session(ts, p["session_start"], p["session_end"]):
        return None

    fast = prev[f"ema_{p['ema_fast']}"]
    slow = prev[f"ema_{p['ema_slow']}"]

    hh = prev[f"hh_{p['smc_bos_lookback']}"]
    ll = prev[f"ll_{p['smc_bos_lookback']}"]

    if pd.isna(hh) or pd.isna(ll):
        return None

    bullish_bias = fast > slow
    bearish_bias = fast < slow

    bullish_bos = row["close"] > hh + p["smc_bos_buffer"]
    bearish_bos = row["close"] < ll - p["smc_bos_buffer"]

    displacement = row["body"] >= row["atr_14"] * p["smc_displacement_body_atr"]
    near_ema = abs(row["close"] - prev["ema_21"]) <= row["atr_14"] * p["smc_pullback_atr"]

    score = 0.0
    if bullish_bias:
        score += 1.5
    if bearish_bias:
        score += 1.5
    if displacement:
        score += 1.5
    if near_ema:
        score += 1.5

    if bullish_bias and bullish_bos and near_ema:
        entry = row["close"] + p["slippage"]
        stop = row["low"] - p["smc_stop_buffer"]
        risk = entry - stop
        if risk <= 0:
            return None
        target = entry + risk * p["smc_rr"]

        return Signal(
            family="smc",
            direction=1,
            score=score,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            reason="smc_long",
        )

    if bearish_bias and bearish_bos and near_ema:
        entry = row["close"] - p["slippage"]
        stop = row["high"] + p["smc_stop_buffer"]
        risk = stop - entry
        if risk <= 0:
            return None
        target = entry - risk * p["smc_rr"]

        return Signal(
            family="smc",
            direction=-1,
            score=score,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            reason="smc_short",
        )

    return None


def choose_best_signal(signals: List[Optional[Signal]], p: Dict) -> Optional[Signal]:
    valid = [s for s in signals if s is not None]
    if not valid:
        return None

    filtered = []
    for s in valid:
        threshold = p[f"{s.family}_min_score"]
        if s.score >= threshold:
            filtered.append(s)

    if not filtered:
        return None

    filtered.sort(key=lambda s: s.score, reverse=True)
    return filtered[0]


def manage_trade(df: pd.DataFrame, signal: Signal, i: int, end_idx: int, p: Dict) -> Trade:
    max_hold = p["global_max_hold_bars"]
    last_idx = min(i + max_hold, end_idx - 1)

    exit_price = None
    exit_reason = None
    exit_idx = None

    for j in range(i + 1, last_idx + 1):
        c = df.iloc[j]
        high = c["high"]
        low = c["low"]

        if signal.direction == 1:
            stop_hit = low <= signal.stop_price
            target_hit = high >= signal.target_price
            if stop_hit:
                exit_price = signal.stop_price
                exit_reason = "stop"
                exit_idx = j
                break
            if target_hit:
                exit_price = signal.target_price
                exit_reason = "target"
                exit_idx = j
                break
        else:
            stop_hit = high >= signal.stop_price
            target_hit = low <= signal.target_price
            if stop_hit:
                exit_price = signal.stop_price
                exit_reason = "stop"
                exit_idx = j
                break
            if target_hit:
                exit_price = signal.target_price
                exit_reason = "target"
                exit_idx = j
                break

    if exit_price is None:
        exit_idx = last_idx
        exit_price = float(df.iloc[last_idx]["close"])
        exit_reason = "timeout"

    if signal.direction == 1:
        result_points = exit_price - signal.entry_price
        risk = signal.entry_price - signal.stop_price
    else:
        result_points = signal.entry_price - exit_price
        risk = signal.stop_price - signal.entry_price

    result_r = result_points / risk if risk > 0 else 0.0

    return Trade(
        family=signal.family,
        entry_idx=i,
        exit_idx=exit_idx,
        direction=signal.direction,
        entry_price=signal.entry_price,
        exit_price=exit_price,
        stop_price=signal.stop_price,
        target_price=signal.target_price,
        result_points=result_points,
        result_r=result_r,
        exit_reason=exit_reason,
        reason=signal.reason,
    )


def run_backtest_multi_setup(df: pd.DataFrame, start_idx: int, end_idx: int, p: Dict) -> List[Trade]:
    trades: List[Trade] = []
    i = max(35, start_idx + 1)

    while i < end_idx - 2:
        old_i = i

        bo = calc_breakout_signal(df, i, p)
        sw = calc_sweep_signal(df, i, p)
        smc = calc_smc_signal(df, i, p)

        best = choose_best_signal([bo, sw, smc], p)

        if best is None:
            i += 1
            continue

        trade = manage_trade(df, best, i, end_idx, p)
        trades.append(trade)
        i = trade.exit_idx + 1

        if i == old_i:
            i += 1

    return trades


def calc_metrics(trades: List[Trade]) -> Metrics:
    if not trades:
        return Metrics(
            trades=0,
            wins=0,
            losses=0,
            timeouts=0,
            win_rate=0.0,
            total_points=0.0,
            total_r=0.0,
            avg_r=0.0,
            expectancy_r=0.0,
            profit_factor=0.0,
            max_dd_r=0.0,
            avg_hold_bars=0.0,
        )

    arr_r = np.array([t.result_r for t in trades], dtype=float)
    arr_p = np.array([t.result_points for t in trades], dtype=float)
    holds = np.array([t.exit_idx - t.entry_idx for t in trades], dtype=float)

    wins = int(np.sum(arr_r > 0))
    losses = int(np.sum(arr_r < 0))
    timeouts = int(sum(1 for t in trades if t.exit_reason == "timeout"))

    gross_profit = float(np.sum(arr_r[arr_r > 0])) if np.any(arr_r > 0) else 0.0
    gross_loss = abs(float(np.sum(arr_r[arr_r < 0]))) if np.any(arr_r < 0) else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    equity = np.cumsum(arr_r)
    peaks = np.maximum.accumulate(equity) if len(equity) else np.array([0.0])
    dd = peaks - equity if len(equity) else np.array([0.0])
    max_dd_r = float(np.max(dd)) if len(dd) else 0.0

    total_r = float(np.sum(arr_r))
    avg_r = float(np.mean(arr_r))
    total_points = float(np.sum(arr_p))
    expectancy_r = avg_r
    win_rate = (wins / len(trades)) * 100.0

    return Metrics(
        trades=len(trades),
        wins=wins,
        losses=losses,
        timeouts=timeouts,
        win_rate=win_rate,
        total_points=total_points,
        total_r=total_r,
        avg_r=avg_r,
        expectancy_r=expectancy_r,
        profit_factor=pf,
        max_dd_r=max_dd_r,
        avg_hold_bars=float(np.mean(holds)) if len(holds) else 0.0,
    )


def robust_score(train_m: Metrics, valid_m: Metrics, test_m: Metrics) -> float:
    if train_m.trades < 8:
        return -2000 + train_m.trades
    if valid_m.trades < 3:
        return -1000 + valid_m.trades
    if test_m.trades < 3:
        return -1000 + test_m.trades

    score = 0.0
    score += min(valid_m.profit_factor, 4.0) * 18.0
    score += min(test_m.profit_factor, 4.0) * 22.0
    score += valid_m.total_r * 1.5
    score += test_m.total_r * 2.0
    score += valid_m.avg_r * 25.0
    score += test_m.avg_r * 30.0
    score += (valid_m.win_rate / 100.0) * 4.0
    score += (test_m.win_rate / 100.0) * 4.0
    score -= valid_m.max_dd_r * 1.5
    score -= test_m.max_dd_r * 2.0

    pf_gap = abs(valid_m.profit_factor - test_m.profit_factor)
    score -= pf_gap * 6.0

    r_gap = abs(valid_m.total_r - test_m.total_r)
    score -= r_gap * 1.0

    return score


def sample_params(rng: random.Random) -> Dict:
    session_start = rng.randint(8, 12)
    session_end = rng.randint(max(13, session_start + 1), 18)
    ema_pairs = [(5, 13), (5, 21), (9, 21), (9, 34), (13, 34), (21, 50)]
    ema_fast, ema_slow = rng.choice(ema_pairs)

    return {
        "use_session": rng.choice([True, False]),
        "session_start": session_start,
        "session_end": session_end,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "slippage": rng.randint(0, 5),
        "global_max_hold_bars": rng.randint(3, 80),

        "bo_lookback": rng.choice([5, 8, 13, 21]),
        "bo_buffer": rng.randint(0, 20),
        "bo_min_range": rng.randint(0, 20),
        "bo_min_body": rng.randint(0, 20),
        "bo_min_atr_ratio": round(rng.uniform(0.0, 1.2), 2),
        "bo_expansion_body_atr": round(rng.uniform(0.0, 1.2), 2),
        "bo_vol_mult": round(rng.uniform(0.0, 1.3), 2),
        "bo_slope_min": round(rng.uniform(0.0, 1.0), 4),
        "bo_require_trend": rng.choice([True, False]),
        "bo_stop_points": rng.randint(10, 150),
        "bo_rr": round(rng.uniform(0.8, 5.0), 2),
        "breakout_min_score": round(rng.uniform(0.0, 4.0), 2),

        "sw_sweep_buffer": rng.randint(0, 15),
        "sw_rejection_body_atr": round(rng.uniform(0.0, 1.0), 2),
        "sw_stop_buffer": rng.randint(1, 20),
        "sw_rr": round(rng.uniform(0.8, 5.0), 2),
        "sweep_min_score": round(rng.uniform(0.0, 4.0), 2),

        "smc_bos_lookback": rng.choice([5, 8, 13, 21]),
        "smc_bos_buffer": rng.randint(0, 15),
        "smc_displacement_body_atr": round(rng.uniform(0.0, 1.2), 2),
        "smc_pullback_atr": round(rng.uniform(0.0, 1.2), 2),
        "smc_stop_buffer": rng.randint(1, 20),
        "smc_rr": round(rng.uniform(0.8, 6.0), 2),
        "smc_min_score": round(rng.uniform(0.0, 5.0), 2),
    }


def worker_init(csv_path: str):
    global G_DF
    base_df = load_csv(csv_path)
    G_DF = add_features(base_df)


def evaluate_candidate(candidate_id: int, params: Dict) -> Dict:
    global G_DF
    df = G_DF

    splits = split_walkforward(len(df), train_pct=0.60, valid_pct=0.20)

    train_trades = run_backtest_multi_setup(df, *splits["train"], params)
    valid_trades = run_backtest_multi_setup(df, *splits["valid"], params)
    test_trades = run_backtest_multi_setup(df, *splits["test"], params)

    train_m = calc_metrics(train_trades)
    valid_m = calc_metrics(valid_trades)
    test_m = calc_metrics(test_trades)

    score = robust_score(train_m, valid_m, test_m)

    return {
        "candidate_id": candidate_id,
        "score": score,
        **params,

        "train_trades": train_m.trades,
        "train_pf": train_m.profit_factor,
        "train_total_r": train_m.total_r,
        "train_avg_r": train_m.avg_r,
        "train_win_rate": train_m.win_rate,
        "train_max_dd_r": train_m.max_dd_r,

        "valid_trades": valid_m.trades,
        "valid_pf": valid_m.profit_factor,
        "valid_total_r": valid_m.total_r,
        "valid_avg_r": valid_m.avg_r,
        "valid_win_rate": valid_m.win_rate,
        "valid_max_dd_r": valid_m.max_dd_r,

        "test_trades": test_m.trades,
        "test_pf": test_m.profit_factor,
        "test_total_r": test_m.total_r,
        "test_avg_r": test_m.avg_r,
        "test_win_rate": test_m.win_rate,
        "test_max_dd_r": test_m.max_dd_r,
    }


def save_csv(rows: List[Dict], path: str):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def save_checkpoint(rows: List[Dict], path: str):
    if not rows:
        return
    pd.DataFrame(rows).to_parquet(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="3-in-1 Hedge Fund Optimizer")
    parser.add_argument("--csv", type=str, default="wdo_m5.csv")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="checkpoint_latest.parquet")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV não encontrado: {args.csv}")

    print("=" * 80)
    print("3-IN-1 HEDGE FUND OPTIMIZER START")
    print(f"CSV: {args.csv}")
    print(f"ITERATIONS: {args.iterations}")
    print(f"WORKERS: {args.workers}")
    print("=" * 80)

    rng = random.Random(args.seed)
    candidates = [(i, sample_params(rng)) for i in range(args.iterations)]

    results: List[Dict] = []
    started = time.time()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(args.csv,),
    ) as ex:
        futures = [ex.submit(evaluate_candidate, cid, params) for cid, params in candidates]

        for idx, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            results.append(row)

            if idx % 10 == 0:
                elapsed = time.time() - started
                rate = idx / elapsed if elapsed > 0 else 0.0
                best_score = max(r["score"] for r in results)
                print(f"[{idx}/{args.iterations}] done | best_score={best_score:.2f} | {rate:.2f} eval/s")

            if idx % args.save_every == 0:
                save_csv(results, "optimizer_results_all.csv")
                pd.DataFrame(results).sort_values("score", ascending=False).head(args.top_k).to_csv(
                    "optimizer_results_top.csv", index=False
                )
                save_checkpoint(results, args.checkpoint)

    save_csv(results, "optimizer_results_all.csv")
    top_df = pd.DataFrame(results).sort_values("score", ascending=False).head(args.top_k)
    top_df.to_csv("optimizer_results_top.csv", index=False)
    save_checkpoint(results, args.checkpoint)

    print("\nTOP 10")
    print(top_df.head(10).to_string(index=False))
    print("\nArquivos salvos:")
    print("- optimizer_results_all.csv")
    print("- optimizer_results_top.csv")
    print(f"- {args.checkpoint}")


if __name__ == "__main__":
    main()
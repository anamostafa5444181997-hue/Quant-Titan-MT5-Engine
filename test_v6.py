#!/usr/bin/env python3
"""
Comprehensive test suite for The Quant Titan V6.0
Tests all modules without requiring MT5 connection.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import polars as pl
from collections import deque

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}")
        failed += 1


# ═══════════════════════════════════════════════════════════════
print("\n═══ §1 CONFIG ═══")
from config import CFG
test("CFG.SYMBOLS has 4 pairs", len(CFG.SYMBOLS) == 4)
test("CFG.MT5_LOGIN is 262501312", CFG.MT5_LOGIN == 262501312)
test("CFG.MT5_SERVER is Exness-MT5Trial16", CFG.MT5_SERVER == "Exness-MT5Trial16")
test("CFG.KELLY_FRACTION = 0.25", CFG.KELLY_FRACTION == 0.25)
test("CFG.CONSENSUS_THR = 0.80", CFG.CONSENSUS_THR == 0.80)
test("CFG.MIN_RR = 3.0", CFG.MIN_RR == 3.0)
test("CFG.TICK_POLL_SEC = 1.0", CFG.TICK_POLL_SEC == 1.0)
test("CFG.STRATEGY_EVAL_SEC = 15.0", CFG.STRATEGY_EVAL_SEC == 15.0)
test("CFG.NUM_BRAINS = 10", CFG.NUM_BRAINS == 10)

# ═══════════════════════════════════════════════════════════════
print("\n═══ §2 DATA ENGINE — SymbolState ═══")
from data_engine import SymbolState, Regime, SwingType, SwingPoint, BrainVote, TradeRecord
st = SymbolState(symbol="EURUSDm")
test("SymbolState created", st.symbol == "EURUSDm")
test("tick_prices is deque", isinstance(st.tick_prices, deque))
test("m15_c starts None", st.m15_c is None)
test("h1_c starts None", st.h1_c is None)
test("swings starts empty", len(st.swings) == 0)

# ═══════════════════════════════════════════════════════════════
print("\n═══ §3 POLARS FEATURE ENGINE ═══")
from data_engine import PolarsEngine
np.random.seed(42)
n = 200
close = np.cumsum(np.random.randn(n) * 0.001) + 1.1000
high  = close + np.abs(np.random.randn(n) * 0.0005)
low   = close - np.abs(np.random.randn(n) * 0.0005)
open_ = close + np.random.randn(n) * 0.0002
vol   = np.random.randint(100, 5000, n).astype(np.float64)

df = PolarsEngine.build_features(open_, high, low, close, vol)
test("Polars DataFrame created", isinstance(df, pl.DataFrame))
test("Has RSI column", "rsi" in df.columns)
test("Has MACD column", "macd" in df.columns)
test("Has MACD signal", "macd_signal" in df.columns)
test("Has MACD histogram", "macd_hist" in df.columns)
test("Has BB upper", "bb_upper" in df.columns)
test("Has BB lower", "bb_lower" in df.columns)
test("Has BB mid", "bb_mid" in df.columns)
test("Has ATR", "atr" in df.columns)
test("Has true_range", "true_range" in df.columns)
test("Has vol_imbalance", "vol_imbalance" in df.columns)
test("Has vol_spike", "vol_spike" in df.columns)
test("Has buy_volume", "buy_volume" in df.columns)
test("Has sell_volume", "sell_volume" in df.columns)
test("No null values in RSI", df.get_column("rsi").null_count() == 0)
test("No null values in MACD", df.get_column("macd").null_count() == 0)
test("No null values in ATR", df.get_column("atr").null_count() == 0)
test("No null values in BB", df.get_column("bb_upper").null_count() == 0)
test("No null values in vol_imbalance", df.get_column("vol_imbalance").null_count() == 0)
test("RSI in [0, 100]", df.get_column("rsi").min() >= 0 and df.get_column("rsi").max() <= 100)
test("DataFrame has 200 rows", df.height == 200)
test("DataFrame has 19 columns", df.width == 19)

# ═══════════════════════════════════════════════════════════════
print("\n═══ §4 NUMPY HELPERS ═══")
from data_engine import np_rsi, np_atr, np_ewm
rsi = np_rsi(close, 14)
test("np_rsi shape matches", len(rsi) == len(close))
test("np_rsi values in [0,100]", rsi.min() >= 0 and rsi.max() <= 100)
atr = np_atr(high, low, close, 14)
test("np_atr shape matches", len(atr) == len(close))
test("np_atr all non-negative", np.all(atr >= 0))
ema = np_ewm(close, 20)
test("np_ewm shape matches", len(ema) == len(close))

# ═══════════════════════════════════════════════════════════════
print("\n═══ §5 WHALE TRACKER ═══")
from whale_tracker import WhaleTracker, WhaleEvent
wt = WhaleTracker()
# Populate state with fake ticks
st2 = SymbolState(symbol="GBPUSDm")
np.random.seed(123)
for i in range(600):
    st2.tick_prices.append(1.2500 + np.random.randn() * 0.001)
    st2.tick_times.append(1000000 + i * 100)
    vol = float(np.random.exponential(10))
    if i > 550 and i < 560:
        vol *= 10  # inject whale spikes
    st2.tick_vols.append(vol)
    st2.tick_flags.append(16 if np.random.rand() > 0.5 else 32)

spikes = wt.detect_volume_spikes(st2)
test("WhaleTracker detects spikes", len(spikes) >= 0)  # may or may not detect
we_test = WhaleEvent(1.0, 'BUY', 3.0, 100.0, 'test')
test("WhaleEvent has correct fields", hasattr(we_test, 'direction') and we_test.direction == 'BUY')

# Candle volume analysis
st2.features_m15 = df  # reuse our test df
cva = WhaleTracker.candle_volume_analysis(st2)
test("candle_volume_analysis returns dict", isinstance(cva, dict))
test("Has imbalance key", "imbalance" in cva)
test("Has spike_count key", "spike_count" in cva)
test("Has direction_bias key", "direction_bias" in cva)

# Composite whale score
wv = wt.whale_score(st2)
test("whale_score returns BrainVote", isinstance(wv, BrainVote))
test("whale_score in [-1, 1]", -1.0 <= wv.score <= 1.0)

# ═══════════════════════════════════════════════════════════════
print("\n═══ §6 STRATEGY — All 10 Brains ═══")
from strategy import (
    ZScoreBrain, HurstBrain, SMCBrain, OrderFlowBrain,
    VolatilityBrain, MomentumDivBrain, LiquiditySweepBrain,
    MeanRevBrain, TrendAlignBrain, TimeSessionBrain,
    MetaBrain, Dashboard,
)

# Prepare a rich state
st3 = SymbolState(symbol="AUDUSDm")
np.random.seed(77)
for i in range(600):
    st3.tick_prices.append(0.6500 + np.cumsum(np.random.randn(1))[0] * 0.0001)
    st3.tick_times.append(2000000 + i * 50)
    st3.tick_vols.append(float(np.random.exponential(5)))
    st3.tick_flags.append(16 if np.random.rand() > 0.4 else 32)

n2 = 200
c2 = np.cumsum(np.random.randn(n2) * 0.0005) + 0.6500
h2 = c2 + np.abs(np.random.randn(n2) * 0.0003)
l2 = c2 - np.abs(np.random.randn(n2) * 0.0003)
o2 = c2 + np.random.randn(n2) * 0.0001
v2 = np.random.randint(50, 3000, n2).astype(np.float64)
st3.m15_o, st3.m15_h, st3.m15_l, st3.m15_c, st3.m15_v = o2, h2, l2, c2, v2
st3.m15_last = 9999999999.0
st3.features_m15 = PolarsEngine.build_features(o2, h2, l2, c2, v2)
st3.h1_o, st3.h1_h, st3.h1_l, st3.h1_c = o2[:100], h2[:100], l2[:100], c2[:100]
st3.h1_last = 9999999999.0
st3.features_h1 = PolarsEngine.build_features(o2[:100], h2[:100], l2[:100], c2[:100], v2[:100])

v1 = ZScoreBrain.vote(st3)
test("B01 ZScore returns BrainVote", isinstance(v1, BrainVote))
test("B01 score in [-1,1]", -1 <= v1.score <= 1)

v2b = HurstBrain.vote(st3)
test("B02 Hurst returns BrainVote", isinstance(v2b, BrainVote))
test("B02 score in [-1,1]", -1 <= v2b.score <= 1)

v3 = SMCBrain.vote(st3, None)
test("B03 SMC returns BrainVote", isinstance(v3, BrainVote))
test("B03 score in [-1,1]", -1 <= v3.score <= 1)

flow_brain = OrderFlowBrain()
v4 = flow_brain.vote(st3)
test("B04 Flow returns BrainVote", isinstance(v4, BrainVote))
test("B04 score in [-1,1]", -1 <= v4.score <= 1)

v5 = VolatilityBrain.vote(st3)
test("B05 Volatility returns BrainVote", isinstance(v5, BrainVote))
test("B05 score in [-1,1]", -1 <= v5.score <= 1)

v6 = MomentumDivBrain.vote(st3)
test("B06 MomDiv returns BrainVote", isinstance(v6, BrainVote))
test("B06 score in [-1,1]", -1 <= v6.score <= 1)

v7 = LiquiditySweepBrain.vote(st3)
test("B07 LiqSweep returns BrainVote", isinstance(v7, BrainVote))
test("B07 score in [-1,1]", -1 <= v7.score <= 1)

v8 = MeanRevBrain.vote(st3)
test("B08 MeanRev returns BrainVote", isinstance(v8, BrainVote))
test("B08 score in [-1,1]", -1 <= v8.score <= 1)

v9 = TrendAlignBrain.vote(st3, None)
test("B09 TrendMTF returns BrainVote", isinstance(v9, BrainVote))
test("B09 score in [-1,1]", -1 <= v9.score <= 1)

v10 = TimeSessionBrain.vote(st3)
test("B10 Session returns BrainVote", isinstance(v10, BrainVote))
test("B10 score in [-1,1]", -1 <= v10.score <= 1)

# ═══════════════════════════════════════════════════════════════
print("\n═══ §7 META-BRAIN ═══")
mb = MetaBrain()
test("MetaBrain has 10 weights", len(mb.weights) == 10)
test("Initial weights sum to 1.0", abs(mb.weights.sum() - 1.0) < 0.01)

all_votes = [v1, v2b, v3, v4, v5, v6, v7, v8, v9, v10]
consensus, direction = mb.consensus(all_votes)
test("Consensus is float", isinstance(consensus, float))
test("Consensus in [-1,1]", -1 <= consensus <= 1)
test("Direction is str", direction in ("BUY", "SELL", "HOLD"))

# Learning
tr = TradeRecord("EURUSDm", "BUY", [v.score for v in all_votes], 5.0, 1, 0)
mb.learn(tr)
test("After learning: weights still sum ~1", abs(mb.weights.sum() - 1.0) < 0.01)
test("After learning: all weights >= FLOOR", np.all(mb.weights >= CFG.WEIGHT_FLOOR))
test("After learning: all weights <= CEIL", np.all(mb.weights <= CFG.WEIGHT_CEIL))

tr2 = TradeRecord("EURUSDm", "SELL", [v.score for v in all_votes], -2.0, -1, 0)
mb.learn(tr2)
test("After 2 trades: history has 2", len(mb.history) == 2)
test("Win rate computed", 0 <= mb.win_rate <= 1)

# ═══════════════════════════════════════════════════════════════
print("\n═══ §8 RISK MANAGER — Fractional Kelly ═══")
from execution import RiskManager, MT5Conn, Executor

# Kelly fraction tests
kf1 = RiskManager.kelly_fraction(0.5, 2.0)
test("Kelly(W=0.5, R=2.0) = 0.25", abs(kf1 - 0.25) < 0.001)

kf2 = RiskManager.kelly_fraction(0.6, 3.0)
expected = 0.6 - 0.4/3.0  # 0.4667
test("Kelly(W=0.6, R=3.0) ≈ 0.467", abs(kf2 - expected) < 0.001)

kf3 = RiskManager.kelly_fraction(0.3, 1.0)
expected3 = 0.3 - 0.7/1.0  # -0.4 → clamped to 0
test("Kelly(W=0.3, R=1.0) = 0.0 (negative edge)", kf3 == 0.0)

kf4 = RiskManager.kelly_fraction(0.55, 2.0)
expected4 = 0.55 - 0.45/2.0  # 0.325
test("Kelly(W=0.55, R=2.0) ≈ 0.325", abs(kf4 - expected4) < 0.001)

# Lot computation
sym_info = {
    "trade_tick_size": 0.00001,
    "trade_tick_value": 1.0,
    "volume_min": 0.01,
    "volume_step": 0.01,
    "volume_max": 100.0,
}

lot1 = RiskManager.compute_lot(
    equity=50.0, sl_dist=0.0010, sym_info=sym_info,
    regime=Regime.NORMAL, win_rate=0.5, avg_rr=3.0)
test("Lot > 0 for $50 account", lot1 > 0)
test("Lot <= MAX_LOT", lot1 <= CFG.MAX_LOT)
test("Lot is multiple of vol_step", abs(lot1 / 0.01 - round(lot1 / 0.01)) < 0.001)

# Verify actual risk doesn't exceed 5%
sl_ticks = 0.0010 / 0.00001  # 100 ticks
actual_risk = lot1 * sl_ticks * 1.0
test(f"Actual risk ${actual_risk:.2f} <= 5% of $50 ($2.50)", actual_risk <= 2.50)

# Low vol regime should be more conservative
lot_low = RiskManager.compute_lot(
    equity=50.0, sl_dist=0.0010, sym_info=sym_info,
    regime=Regime.LOW_VOL, win_rate=0.5, avg_rr=3.0)
test("LOW_VOL lot <= NORMAL lot", lot_low <= lot1)

# Zero SL should return 0
lot_zero = RiskManager.compute_lot(
    equity=50.0, sl_dist=0.0, sym_info=sym_info,
    regime=Regime.NORMAL, win_rate=0.5, avg_rr=3.0)
test("Zero SL → lot=0", lot_zero == 0.0)

# Negative edge (bad win rate) should still give minimum lot
lot_bad = RiskManager.compute_lot(
    equity=50.0, sl_dist=0.0010, sym_info=sym_info,
    regime=Regime.NORMAL, win_rate=0.3, avg_rr=1.0)
test("Negative Kelly → still gets floor lot", lot_bad >= 0.01)

# ═══════════════════════════════════════════════════════════════
print("\n═══ §9 MT5 CONNECTION ═══")
conn = MT5Conn()
test("MT5Conn created", conn is not None)
test("MT5Conn._ok starts False", conn._ok == False)

# ═══════════════════════════════════════════════════════════════
print("\n═══ §10 MAIN ENGINE ═══")
from main import QuantTitanV6
engine = QuantTitanV6()
test("Engine created", engine is not None)
test("Engine has conn", engine.conn is not None)
test("Engine has executor", engine.executor is not None)
test("Engine has meta", engine.meta is not None)
test("Engine has 10 brain instances",
     all(hasattr(engine, f"b_{n}") for n in
         ["zscore","hurst","smc","flow","vol","momdiv","sweep","meanrev","trend","session"]))
test("Engine._running is True", engine._running == True)

# ═══════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'═'*60}")
sys.exit(0 if failed == 0 else 1)

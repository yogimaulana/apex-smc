# main.py - Apex SMC Intelligence v12.5
# Zona high-prob (OB/FVG) = boleh sinyal | OTE = bonus | fix MISS | news warning | max SL $8

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import time, os, json, requests, numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SMC-Engine")

app = FastAPI(title="Apex SMC Intelligence", version="12.5")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

HISTORY_FILE = "trade_history.json"
LEARNING_FILE = "learning_stats.json"
CACHE_TTL = 15
FUND_CACHE_TTL = 300
MIN_SCORE_BASE = 76
ENTRY_ZONE_ATR_FACTOR = 0.40

MAX_SL_USD = 8.00
MIN_SL_USD = 1.20
TP1_RR = 1.6
TP2_RR = 2.2
ENTRY_ATR_FALLBACK = 0.12
OTE_LOW, OTE_HIGH = 0.62, 0.79

INTERVAL_MAP = {
    "1min": "1m", "1m": "1m", "5min": "5m", "5m": "5m",
    "15min": "15m", "15m": "15m", "30min": "30m", "30m": "30m",
    "1h": "1h", "60min": "1h", "4h": "4h", "1day": "1d", "1d": "1d",
}

cache_store = {}
price_cache = {}
fundamental_cache = {"data": None, "time": 0}
trade_history: List[Dict] = []
learning_stats: Dict = {
    "total_closed": 0, "tp_hits": 0, "sl_hits": 0, "miss_entries": 0,
    "score_adjust": 0, "by_pattern": {}, "recent_results": [],
    "blocked_patterns": {}, "last_update": None
}

def norm_symbol(s: str) -> str:
    return (s or "").replace("/", "").replace("'", "").strip().upper()

def load_history():
    global trade_history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                trade_history = json.load(f)
    except:
        trade_history = []

def save_history():
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(trade_history[:120], f, indent=2)
    except Exception as e:
        logger.error(e)

def load_learning():
    global learning_stats
    try:
        if os.path.exists(LEARNING_FILE):
            with open(LEARNING_FILE, "r") as f:
                learning_stats.update(json.load(f))
    except:
        pass

def save_learning():
    try:
        learning_stats["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LEARNING_FILE, "w") as f:
            json.dump(learning_stats, f, indent=2)
    except Exception as e:
        logger.error(e)

load_history()
load_learning()

def fetch_realtime_price(symbol: str) -> Dict:
    key = f"price_{norm_symbol(symbol)}"
    now = time.time()
    if key in price_cache and (now - price_cache[key]["time"]) < 3:
        return price_cache[key]["data"]
    try:
        clean = norm_symbol(symbol)
        r = requests.get(f"https://biquote.io/api/{clean}", timeout=8)
        data = r.json()
        if "mid" in data:
            result = {
                "symbol": clean, "price": float(data["mid"]),
                "bid": float(data.get("bid", data["mid"])),
                "ask": float(data.get("ask", data["mid"])),
                "spread": float(data.get("spread", 0)),
                "timestamp": data.get("timestamp") or data.get("lastQuoteAt"),
                "source": data.get("source", "biquote-MT5"), "stale": data.get("stale", False)
            }
            price_cache[key] = {"time": now, "data": result}
            return result
    except Exception as e:
        logger.error(f"biquote price error: {e}")
    if key in price_cache:
        return price_cache[key]["data"]
    return {"symbol": norm_symbol(symbol), "price": 0, "source": "error", "stale": True}

def fetch_candles(symbol: str, interval: str = "5min", outputsize: int = 100, force: bool = False) -> List[Dict]:
    sym = norm_symbol(symbol)
    bq_interval = INTERVAL_MAP.get(interval, interval)
    key = f"{sym}_{bq_interval}"
    now = time.time()
    if not force and key in cache_store and (now - cache_store[key]["time"]) < CACHE_TTL:
        return cache_store[key]["data"]
    try:
        url = f"https://biquote.io/api/{sym}/ohlc?interval={bq_interval}&limit={outputsize}"
        r = requests.get(url, timeout=12)
        data = r.json()
        bars = data.get("bars") or []
        if not bars:
            return cache_store.get(key, {}).get("data", [])
        candles = []
        for b in reversed(bars):
            candles.append({
                "datetime": b.get("openTime") or b.get("time") or "",
                "open": float(b["open"]), "high": float(b["high"]),
                "low": float(b["low"]), "close": float(b["close"]),
                "volume": float(b.get("volume") or b.get("tickVolume") or 0)
            })
        cache_store[key] = {"time": now, "data": candles}
        return candles
    except Exception as e:
        logger.error(f"biquote ohlc error: {e}")
        return cache_store.get(key, {}).get("data", [])

def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return 1.5
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return round(float(np.mean(trs[-period:])), 5)

def find_swings(highs, lows, left=3, right=3):
    sh, sl = [], []
    n = len(highs)
    for i in range(left, n - right):
        if highs[i] == max(highs[i - left:i + right + 1]):
            sh.append((i, float(highs[i])))
        if lows[i] == min(lows[i - left:i + right + 1]):
            sl.append((i, float(lows[i])))
    return sh, sl

def get_structure_bias(candles) -> Tuple[str, str]:
    if len(candles) < 25:
        return "NEUTRAL", "Insufficient Data"
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    closes = np.array([c["close"] for c in candles])
    current = float(closes[-1])
    sh, sl = find_swings(highs, lows, 3, 3)
    if len(sh) < 2 or len(sl) < 2:
        sh, sl = find_swings(highs, lows, 2, 2)
        if len(sh) < 2 or len(sl) < 2:
            return "NEUTRAL", "RANGE / CHOPPY"
    last_sh = sh[-3:] if len(sh) >= 3 else sh
    last_sl = sl[-3:] if len(sl) >= 3 else sl
    bias, structure = "NEUTRAL", "RANGE / CHOPPY"
    if len(last_sh) >= 2 and len(last_sl) >= 2:
        if last_sh[-1][1] > last_sh[-2][1] and last_sl[-1][1] > last_sl[-2][1]:
            structure, bias = "BULLISH BOS (HH+HL)", "BULLISH"
        elif last_sh[-1][1] < last_sh[-2][1] and last_sl[-1][1] < last_sl[-2][1]:
            structure, bias = "BEARISH BOS (LH+LL)", "BEARISH"
        elif last_sh[-1][1] > last_sh[-2][1] and last_sl[-1][1] < last_sl[-2][1]:
            structure, bias = "CHoCH → Bullish", "BULLISH"
        elif last_sh[-1][1] < last_sh[-2][1] and last_sl[-1][1] > last_sl[-2][1]:
            structure, bias = "CHoCH → Bearish", "BEARISH"
    recent_high = max(s[1] for s in sh[-4:]) if len(sh) >= 2 else float(highs[-5])
    recent_low = min(s[1] for s in sl[-4:]) if len(sl) >= 2 else float(lows[-5])
    if current < recent_low and bias != "BEARISH":
        if len(sh) >= 2 and sh[-1][1] < sh[-2][1]:
            structure, bias = "CHoCH → Bearish (Break Low)", "BEARISH"
        elif bias == "NEUTRAL":
            structure, bias = "BEARISH BOS (Break Low)", "BEARISH"
    if current > recent_high and bias != "BULLISH":
        if len(sl) >= 2 and sl[-1][1] > sl[-2][1]:
            structure, bias = "CHoCH → Bullish (Break High)", "BULLISH"
        elif bias == "NEUTRAL":
            structure, bias = "BULLISH BOS (Break High)", "BULLISH"
    return bias, structure

def swing_range(candles) -> Optional[Tuple[float, float]]:
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    sh, sl = find_swings(highs, lows, 2, 2)
    if len(sh) < 1 or len(sl) < 1:
        hi = float(np.max(highs[-25:]))
        lo = float(np.min(lows[-25:]))
    else:
        hi = max(sh[-1][1], float(np.max(highs[-20:])))
        lo = min(sl[-1][1], float(np.min(lows[-20:])))
    if hi - lo < 1.0:
        return None
    return lo, hi

def parse_zone(text: str) -> Optional[Tuple[float, float]]:
    if not text or text in ("None", "No Valid FVG"):
        return None
    m = re.search(r"([\d.]+)\s*[-–]\s*([\d.]+)", text)
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    return (min(a, b), max(a, b))

def build_execution_levels(bias: str, current: float, atr: float, ob_str: str, fvg_str: str, candles: List[Dict]) -> Optional[Dict]:
    """
    Wajib: zona OB atau FVG valid (high probability).
    OTE = bonus (in_ote=True), BUKAN syarat.
    SL struktur + CAP $8. Anti miss instan (TP di sisi benar vs harga).
    """
    is_buy = bias == "BULLISH"
    zone = None
    if is_buy and "Bullish OB" in (ob_str or ""):
        zone = parse_zone(ob_str)
    elif not is_buy and "Bearish OB" in (ob_str or ""):
        zone = parse_zone(ob_str)
    if zone is None:
        if is_buy and "Bullish FVG" in (fvg_str or ""):
            zone = parse_zone(fvg_str)
        elif not is_buy and "Bearish FVG" in (fvg_str or ""):
            zone = parse_zone(fvg_str)
    if zone is None:
        return None

    # OTE check (bonus only)
    in_ote = False
    sr = swing_range(candles)
    if sr:
        lo, hi = sr
        rng = hi - lo
        mid_z = (zone[0] + zone[1]) / 2
        if is_buy:
            ote_lo, ote_hi = hi - OTE_HIGH * rng, hi - OTE_LOW * rng
            in_ote = ote_lo <= mid_z <= ote_hi
        else:
            ote_lo, ote_hi = lo + OTE_LOW * rng, lo + OTE_HIGH * rng
            in_ote = ote_lo <= mid_z <= ote_hi

    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    sh, sl_pts = find_swings(np.array(highs), np.array(lows), 2, 2)

    entry = round((zone[0] + zone[1]) / 2, 2)

    if is_buy:
        if entry >= current:
            entry = round(min(zone[1], current - atr * ENTRY_ATR_FALLBACK), 2)
        if entry >= current:
            return None
        struct_sl = zone[0] - atr * 0.15
        if sl_pts:
            struct_sl = min(struct_sl, sl_pts[-1][1] - atr * 0.1)
        raw_dist = abs(entry - struct_sl)
        if raw_dist > MAX_SL_USD:
            return None
        sl_dist = max(MIN_SL_USD, min(raw_dist, MAX_SL_USD))
        sl = round(entry - sl_dist, 2)
        tp1 = round(entry + sl_dist * TP1_RR, 2)
        tp2 = round(entry + sl_dist * TP2_RR, 2)
        if tp1 <= current:
            return None
    else:
        if entry <= current:
            entry = round(max(zone[0], current + atr * ENTRY_ATR_FALLBACK), 2)
        if entry <= current:
            return None
        struct_sl = zone[1] + atr * 0.15
        if sh:
            struct_sl = max(struct_sl, sh[-1][1] + atr * 0.1)
        raw_dist = abs(struct_sl - entry)
        if raw_dist > MAX_SL_USD:
            return None
        sl_dist = max(MIN_SL_USD, min(raw_dist, MAX_SL_USD))
        sl = round(entry + sl_dist, 2)
        tp1 = round(entry - sl_dist * TP1_RR, 2)
        tp2 = round(entry - sl_dist * TP2_RR, 2)
        if tp1 >= current:
            return None

    return {
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
        "rrr": f"1:{TP1_RR}", "sl_dist": round(sl_dist, 2),
        "in_ote": in_ote,
        "quality": "OTE+ZONE" if in_ote else "ZONE"
    }

def make_pattern_key(smc: dict) -> str:
    bias = smc.get("bias", "N")
    htf = smc.get("htf_bias", "N")
    cq = smc.get("confluence_quality", "poor")
    ob = "OB1" if smc.get("order_block") and smc.get("order_block") != "None" else "OB0"
    fvg = "FVG1" if smc.get("fvg") and "No Valid" not in str(smc.get("fvg", "")) else "FVG0"
    liq = "SWEEP" if "Swept" in str(smc.get("liquidity", "")) else "NOSWEEP"
    return f"{bias}|HTF:{htf}|{cq}|{ob}|{fvg}|{liq}"

def record_trade_result(trade: Dict):
    global learning_stats
    status = trade.get("status", "")
    pattern = trade.get("pattern_key", "UNKNOWN")
    learning_stats["total_closed"] = learning_stats.get("total_closed", 0) + 1
    if "by_pattern" not in learning_stats:
        learning_stats["by_pattern"] = {}
    if pattern not in learning_stats["by_pattern"]:
        learning_stats["by_pattern"][pattern] = {"tp": 0, "sl": 0, "miss": 0, "last_result": None}
    if "TP1 HIT" in status or "TP HIT" in status:
        learning_stats["tp_hits"] = learning_stats.get("tp_hits", 0) + 1
        learning_stats["by_pattern"][pattern]["tp"] += 1
        learning_stats["by_pattern"][pattern]["last_result"] = "TP"
        result = "TP"
    elif "SL HIT" in status:
        learning_stats["sl_hits"] = learning_stats.get("sl_hits", 0) + 1
        learning_stats["by_pattern"][pattern]["sl"] += 1
        learning_stats["by_pattern"][pattern]["last_result"] = "SL"
        result = "SL"
    elif "MISS ENTRY" in status or "INVALIDATED" in status or "INVALID SETUP" in status:
        learning_stats["miss_entries"] = learning_stats.get("miss_entries", 0) + 1
        learning_stats["by_pattern"][pattern]["miss"] += 1
        learning_stats["by_pattern"][pattern]["last_result"] = "MISS"
        result = "MISS"
    else:
        return
    recent = learning_stats.get("recent_results", [])
    recent.insert(0, {"pattern": pattern, "result": result, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    learning_stats["recent_results"] = recent[:20]
    pstats = learning_stats["by_pattern"][pattern]
    if result == "SL" and pstats["sl"] >= 2:
        last_two = [r for r in recent if r["pattern"] == pattern][:2]
        if len(last_two) >= 2 and all(r["result"] == "SL" for r in last_two):
            learning_stats.setdefault("blocked_patterns", {})[pattern] = 3
    save_learning()

def apply_learning_to_score(base_score: int, pattern_key: str) -> Tuple[int, str]:
    score = base_score + int(learning_stats.get("score_adjust", 0))
    notes = []
    blocked = learning_stats.get("blocked_patterns", {})
    if pattern_key in blocked and blocked[pattern_key] > 0:
        return 0, "PATTERN_BLOCKED_RECENT_SL"
    p = learning_stats.get("by_pattern", {}).get(pattern_key)
    if p:
        total = p["tp"] + p["sl"]
        if total >= 3:
            wr = p["tp"] / total
            if wr >= 0.65:
                score += 8
                notes.append("pattern_strong")
            elif wr <= 0.35:
                score -= 12
                notes.append("pattern_weak")
    return max(0, min(98, int(score))), (",".join(notes) if notes else "neutral")

def get_dynamic_min_score() -> int:
    adj = int(learning_stats.get("score_adjust", 0))
    return max(70, min(84, MIN_SCORE_BASE - adj))

def is_same_entry_zone(symbol: str, new_entry: float, atr: float) -> bool:
    sym = norm_symbol(symbol)
    for t in trade_history:
        if norm_symbol(t.get("symbol", "")) != sym:
            continue
        try:
            if abs(new_entry - float(t["entry"])) < (atr * ENTRY_ZONE_ATR_FACTOR):
                return True
        except:
            continue
    return False

def get_active_trade(symbol: str) -> Optional[Dict]:
    sym = norm_symbol(symbol)
    return next(
        (t for t in trade_history
         if norm_symbol(t.get("symbol", "")) == sym
         and t.get("status") in ["PENDING ENTRY", "FILLED & ACTIVE"]),
        None
    )

def analyze_pure_smc(candles, htf_candles=None):
    if len(candles) < 30:
        return {
            "bias": "NEUTRAL", "bos_choch": "Insufficient Data", "order_block": "None",
            "fvg": "None", "liquidity": "Consolidating", "score": 30, "setup": "WAIT",
            "current_price": candles[-1]["close"] if candles else 0, "atr": 1.5,
            "last_candle_time": candles[-1]["datetime"] if candles else None,
            "entry_reason": "Insufficient data", "score_detail": {}, "score_label": "LOW",
            "htf_bias": "NEUTRAL", "confluence_quality": "poor", "pattern_key": "NONE",
            "learn_note": "neutral"
        }

    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    current_price = float(closes[-1])
    atr = calculate_atr(candles)
    last_time = candles[-1]["datetime"]

    bias, structure = get_structure_bias(candles)
    htf_bias = "NEUTRAL"
    if htf_candles and len(htf_candles) >= 30:
        htf_bias, _ = get_structure_bias(htf_candles)

    rh = np.max(highs[-15:-2]) if len(highs) > 15 else np.max(highs[:-1])
    rl = np.min(lows[-15:-2]) if len(lows) > 15 else np.min(lows[:-1])
    liq = "Protected"
    if highs[-1] > rh and current_price < rh * 0.998:
        liq = "BSL Swept"
        if bias != "BULLISH":
            bias = "BEARISH"
    elif lows[-1] < rl and current_price > rl * 1.002:
        liq = "SSL Swept"
        if bias != "BEARISH":
            bias = "BULLISH"

    bull_fvg = bear_fvg = None
    for i in range(len(candles) - 3, max(len(candles) - 12, 2), -1):
        if candles[i]["low"] > candles[i - 2]["high"] and bull_fvg is None:
            bull_fvg = f"Bullish FVG [{candles[i-2]['high']:.2f}-{candles[i]['low']:.2f}]"
        if candles[i]["high"] < candles[i - 2]["low"] and bear_fvg is None:
            bear_fvg = f"Bearish FVG [{candles[i]['high']:.2f}-{candles[i-2]['low']:.2f}]"
    if bias == "BULLISH" and bull_fvg:
        fvg = bull_fvg
    elif bias == "BEARISH" and bear_fvg:
        fvg = bear_fvg
    else:
        fvg = bull_fvg or bear_fvg or "No Valid FVG"

    ob = "None"
    for i in range(len(candles) - 3, max(len(candles) - 25, 2), -1):
        if bias == "BULLISH" and candles[i]["close"] < candles[i]["open"]:
            if i + 1 < len(candles) and candles[i + 1]["close"] > candles[i]["high"]:
                ob = f"Bullish OB @ {candles[i]['low']:.2f}-{candles[i]['high']:.2f}"
                break
        if bias == "BEARISH" and candles[i]["close"] > candles[i]["open"]:
            if i + 1 < len(candles) and candles[i + 1]["close"] < candles[i]["low"]:
                ob = f"Bearish OB @ {candles[i]['low']:.2f}-{candles[i]['high']:.2f}"
                break

    fvg_aligned = (bias == "BULLISH" and "Bullish FVG" in fvg) or (bias == "BEARISH" and "Bearish FVG" in fvg)
    ob_aligned = (bias == "BULLISH" and "Bullish OB" in ob) or (bias == "BEARISH" and "Bearish OB" in ob)
    has_structure = "BOS" in structure or "CHoCH" in structure
    htf_conflict = (bias == "BULLISH" and htf_bias == "BEARISH") or (bias == "BEARISH" and htf_bias == "BULLISH")

    if has_structure and (ob_aligned or fvg_aligned) and not htf_conflict and bias != "NEUTRAL":
        confluence_quality = "clean" if (ob_aligned and fvg_aligned) or (ob_aligned and "Swept" in liq) else "partial"
    elif bias != "NEUTRAL" and (ob_aligned or fvg_aligned) and has_structure:
        confluence_quality = "partial"
    else:
        confluence_quality = "poor"

    reasons = []
    if has_structure:
        reasons.append(structure)
    if ob != "None":
        reasons.append(ob)
    if "FVG" in fvg and "No Valid" not in fvg:
        reasons.append(fvg)
    if "Swept" in liq:
        reasons.append(liq)
    if htf_bias != "NEUTRAL":
        reasons.append(f"HTF {htf_bias}")
    entry_reason = " + ".join(reasons) if reasons else "No clear confluence"

    score = 0
    score_detail = {}
    if "BOS" in structure:
        score += 26
        score_detail["structure"] = 26
    elif "CHoCH" in structure:
        score += 22
        score_detail["structure"] = 22
    else:
        score += 4
        score_detail["structure"] = 4
    score_detail["order_block"] = 20 if ob_aligned else 0
    score += score_detail["order_block"]
    score_detail["fvg"] = 16 if fvg_aligned else 0
    score += score_detail["fvg"]
    score_detail["liquidity"] = 12 if "Swept" in liq else 0
    score += score_detail["liquidity"]
    if htf_bias == bias and bias != "NEUTRAL":
        score_detail["htf_alignment"] = 16
    elif htf_conflict:
        score_detail["htf_alignment"] = -12
    else:
        score_detail["htf_alignment"] = 4
    score += score_detail["htf_alignment"]
    score_detail["confluence_bonus"] = 10 if confluence_quality == "clean" else (4 if confluence_quality == "partial" else 0)
    score += score_detail["confluence_bonus"]

    smc_tmp = {"bias": bias, "htf_bias": htf_bias, "confluence_quality": confluence_quality,
               "order_block": ob, "fvg": fvg, "liquidity": liq}
    pattern_key = make_pattern_key(smc_tmp)
    learned_score, learn_note = apply_learning_to_score(score, pattern_key)
    score = learned_score
    score_detail["learning"] = learn_note
    min_sc = get_dynamic_min_score()
    score_label = "HIGH" if score >= 84 else ("MEDIUM" if score >= min_sc else "LOW")

    setup = "WAIT FOR MITIGATION"
    if learn_note == "PATTERN_BLOCKED_RECENT_SL":
        setup = "WAIT - LEARNING BLOCK"
    elif htf_conflict:
        setup = "WAIT - HTF CONFLICT"
    elif not has_structure:
        setup = "WAIT - NO CLEAR STRUCTURE"
    elif bias == "BULLISH" and score >= min_sc and confluence_quality in ("clean", "partial") and (ob_aligned or fvg_aligned):
        setup = "BUY LIMIT (ZONE)"
    elif bias == "BEARISH" and score >= min_sc and confluence_quality in ("clean", "partial") and (ob_aligned or fvg_aligned):
        setup = "SELL LIMIT (ZONE)"

    return {
        "bias": bias, "bos_choch": structure, "order_block": ob, "fvg": fvg,
        "liquidity": liq, "score": score, "score_label": score_label,
        "score_detail": score_detail, "setup": setup,
        "current_price": current_price, "atr": atr, "last_candle_time": last_time,
        "entry_reason": entry_reason, "htf_bias": htf_bias,
        "confluence_quality": confluence_quality, "pattern_key": pattern_key,
        "learn_note": learn_note
    }

def fetch_high_impact():
    now = time.time()
    if fundamental_cache["data"] and (now - fundamental_cache["time"]) < FUND_CACHE_TTL:
        return fundamental_cache["data"]
    try:
        r = requests.get("https://biquote.io/api/calendar/upcoming?countries=US&importance=high", timeout=8)
        events = r.json()
        if isinstance(events, list):
            fundamental_cache["data"] = events
            fundamental_cache["time"] = now
            return events
    except Exception as e:
        logger.error(f"calendar error: {e}")
    return fundamental_cache["data"] or []

def analyze_fundamental_xau():
    events = fetch_high_impact()
    now = datetime.now(timezone.utc)
    critical = ["nonfarm", "nfp", "payroll", "cpi", "core cpi", "pce", "fomc", "interest rate",
                "rate decision", "fed", "adp", "gdp"]
    next_ev, min_m, upcoming = None, 99999, []
    for ev in events:
        try:
            et = datetime.fromisoformat(ev["time"].replace("Z", "+00:00"))
            mins = (et - now).total_seconds() / 60
            if mins < -45:
                continue
            name = ev["name"]
            if any(k in name.lower() for k in critical) or ev.get("importance") == "high":
                info = {"name": name, "minutes_left": round(mins, 1)}
                upcoming.append(info)
                if 0 <= mins < min_m:
                    min_m, next_ev = mins, info
        except:
            continue
    if not next_ev:
        return {
            "scalping_status": "SAFE FOR SCALPING", "risk_level": "LOW",
            "recommendation": "Tidak ada High Impact. Scalping aman.",
            "next_high_impact": "None", "minutes_until_next": None,
            "upcoming_events": [], "warning_only": True
        }
    if next_ev["minutes_left"] <= 60:
        return {
            "scalping_status": "⚠ WARNING - HIGH IMPACT SOON", "risk_level": "HIGH",
            "recommendation": f"PERINGATAN: {next_ev['name']} dalam {next_ev['minutes_left']} menit. Sinyal tetap aktif — kelola risk manual.",
            "next_high_impact": next_ev["name"], "minutes_until_next": next_ev["minutes_left"],
            "upcoming_events": upcoming[:4], "warning_only": True
        }
    if next_ev["minutes_left"] <= 120:
        return {
            "scalping_status": "⚠ CAUTION - NEWS WITHIN 2H", "risk_level": "MEDIUM",
            "recommendation": f"Hati-hati: {next_ev['name']} dalam {round(next_ev['minutes_left']/60, 1)} jam. Sinyal tidak diblokir.",
            "next_high_impact": next_ev["name"], "minutes_until_next": next_ev["minutes_left"],
            "upcoming_events": upcoming[:4], "warning_only": True
        }
    return {
        "scalping_status": "SAFE FOR SCALPING", "risk_level": "LOW",
        "recommendation": f"Aman. Next event masih {round(next_ev['minutes_left']/60, 1)} jam lagi.",
        "next_high_impact": next_ev["name"], "minutes_until_next": next_ev["minutes_left"],
        "upcoming_events": upcoming[:4], "warning_only": True
    }

def manage_trades(symbol: str, current_price: float, current_bias: Optional[str] = None):
    global trade_history
    changed = False
    sym = norm_symbol(symbol)
    for t in trade_history:
        if norm_symbol(t.get("symbol", "")) != sym:
            continue
        entry, sl, tp1 = float(t["entry"]), float(t["sl"]), float(t["tp1"])
        is_buy = "BUY" in t["type"]

        if t["status"] == "PENDING ENTRY":
            if current_bias and current_bias != "NEUTRAL":
                if is_buy and current_bias == "BEARISH":
                    t["status"] = "CLOSED - INVALIDATED"
                    t["close_price"] = str(round(current_price, 2))
                    t["close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    changed = True
                    record_trade_result(t)
                    continue
                if not is_buy and current_bias == "BULLISH":
                    t["status"] = "CLOSED - INVALIDATED"
                    t["close_price"] = str(round(current_price, 2))
                    t["close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    changed = True
                    record_trade_result(t)
                    continue

            filled = (is_buy and current_price <= entry) or (not is_buy and current_price >= entry)
            if is_buy:
                missed = (not filled) and (current_price >= tp1) and (entry < tp1)
            else:
                missed = (not filled) and (current_price <= tp1) and (entry > tp1)

            create_px = float(t.get("create_price") or entry)
            if is_buy and missed and create_px >= tp1:
                t["status"] = "CLOSED - INVALID SETUP"
                t["close_price"] = str(round(current_price, 2))
                t["close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                record_trade_result(t)
                continue
            if not is_buy and missed and create_px <= tp1:
                t["status"] = "CLOSED - INVALID SETUP"
                t["close_price"] = str(round(current_price, 2))
                t["close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                record_trade_result(t)
                continue

            if filled:
                t["status"] = "FILLED & ACTIVE"
                t["fill_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                t["fill_price"] = str(round(current_price, 2))
                changed = True
            elif missed:
                t["status"] = "CLOSED - MISS ENTRY"
                t["close_price"] = str(round(current_price, 2))
                t["close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                record_trade_result(t)

        elif t["status"] == "FILLED & ACTIVE":
            hit = None
            if is_buy:
                if current_price <= sl:
                    hit = "SL HIT"
                elif current_price >= tp1:
                    hit = "TP1 HIT"
            else:
                if current_price >= sl:
                    hit = "SL HIT"
                elif current_price <= tp1:
                    hit = "TP1 HIT"
            if hit:
                t["status"] = f"CLOSED - {hit}"
                t["close_price"] = str(round(current_price, 2))
                t["close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                record_trade_result(t)
    if changed:
        save_history()
    return changed

@app.get("/api/price/{symbol:path}")
async def get_price(symbol: str):
    return fetch_realtime_price(symbol)

@app.get("/api/max-intelligence-signal/{symbol:path}")
async def get_signal(
    symbol: str,
    timeframe: str = Query("5min"),
    custom_atr: Optional[float] = Query(None),
    auto_lock: bool = Query(True)
):
    decoded = norm_symbol(symbol)
    if not decoded:
        raise HTTPException(400, "Symbol required")

    price_data = fetch_realtime_price(decoded)
    current_price = price_data["price"]
    candles = fetch_candles(decoded, timeframe, outputsize=100, force=True)
    if not candles:
        candles = fetch_candles(decoded, timeframe, force=False)
    if not candles:
        raise HTTPException(503, detail="Market data unavailable (biquote OHLC).")
    if current_price > 0:
        candles[-1]["close"] = current_price

    htf_candles = fetch_candles(decoded, "1h", outputsize=80, force=False)
    last_candle_time = candles[-1]["datetime"]
    smc_preview = analyze_pure_smc(candles, htf_candles)
    manage_trades(decoded, current_price, current_bias=smc_preview.get("bias"))

    active = get_active_trade(decoded)
    if active:
        fund = analyze_fundamental_xau()
        return {
            "symbol": decoded, "timeframe": timeframe,
            "market_structure": f"Trade Active: {active['status']}",
            "technical_layer": {
                "order_block": active.get("order_block", "-"),
                "fair_value_gap": "-", "liquidity_pool": "-",
                "institutional_bias": "HOLD" if active["status"] == "FILLED & ACTIVE" else "WAITING ENTRY",
                "entry_reason": active.get("entry_reason", "-"),
                "htf_bias": active.get("htf_bias", "-"),
                "score_label": active.get("score_label", "-"),
                "confluence_quality": active.get("confluence_quality", "-")
            },
            "fundamental_layer": fund,
            "master_decision": {
                "action": f"{'HOLD' if active['status'] == 'FILLED & ACTIVE' else 'WAITING'} ({active['type']})",
                "confidence_score": "100%", "execution_status": active["status"]
            },
            "execution_parameters": {
                "entry_price": active["entry"], "stop_loss": active["sl"],
                "take_profit_1": active["tp1"], "take_profit_2": active["tp2"],
                "risk_to_reward_ratio": active["rrr"], "atr_used": active.get("atr_used", "-"),
                "current_price": str(round(current_price, 2)),
                "price_source": price_data.get("source", "biquote"),
                "last_candle_time": last_candle_time,
                "entry_reason": active.get("entry_reason", "-")
            },
            "learning": {
                "score_adjust": learning_stats.get("score_adjust", 0),
                "tp_hits": learning_stats.get("tp_hits", 0),
                "sl_hits": learning_stats.get("sl_hits", 0),
                "total_closed": learning_stats.get("total_closed", 0)
            },
            "ai_rationale": f"Pair: {decoded} | Status: {active['status']}"
        }

    smc = smc_preview
    fund = analyze_fundamental_xau()
    atr_val = custom_atr if custom_atr and custom_atr > 0 else smc["atr"]
    min_sc = get_dynamic_min_score()

    action = smc["setup"]
    confidence = smc["score"]
    status = "IDLE"
    entry = sl = tp1 = tp2 = "-"
    rrr = "0.0"
    zone_quality = ""

    # News = warning only (tidak block)

    if "HTF CONFLICT" in action or "NO CLEAR STRUCTURE" in action or "LEARNING BLOCK" in action:
        status = action.replace("WAIT - ", "")
    elif smc["score"] < min_sc or smc["confluence_quality"] == "poor":
        action = "WAIT - LOW CONFIDENCE"
        status = "SCORE TOO LOW"
    elif "BUY" in action or "SELL" in action:
        levels = build_execution_levels(
            smc["bias"], current_price, atr_val,
            smc["order_block"], smc["fvg"], candles
        )
        if levels is None:
            action = "WAIT - NO VALID ZONE / RISK"
            status = "ZONE OR RISK REJECT"
        else:
            entry, sl, tp1, tp2 = levels["entry"], levels["sl"], levels["tp1"], levels["tp2"]
            rrr = levels["rrr"]
            zone_quality = levels.get("quality", "ZONE")
            if levels.get("in_ote"):
                confidence = min(98, confidence + 6)
                action = action.replace("(ZONE)", "(OTE+ZONE)")
            if is_same_entry_zone(decoded, float(entry), atr_val):
                action, status = "WAIT - SAME ENTRY ZONE", "DUPLICATE ZONE"
                entry = sl = tp1 = tp2 = "-"
            else:
                status = "PENDING ENTRY" if auto_lock else "SIGNAL READY"

    if status == "PENDING ENTRY":
        new_trade = {
            "id": len(trade_history) + 1,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": decoded, "type": action,
            "entry": str(entry), "sl": str(sl), "tp1": str(tp1), "tp2": str(tp2),
            "rrr": rrr, "status": status, "atr_used": atr_val,
            "create_price": str(round(current_price, 2)),
            "zone_quality": zone_quality,
            "order_block": smc["order_block"], "entry_reason": smc["entry_reason"],
            "bias": smc["bias"], "htf_bias": smc["htf_bias"],
            "score": smc["score"], "score_label": smc["score_label"],
            "confluence_quality": smc["confluence_quality"],
            "pattern_key": smc["pattern_key"]
        }
        trade_history.insert(0, new_trade)
        save_history()

    rationale = (
        f"v12.5 | {decoded} | Score {confidence} | {smc['confluence_quality']} | "
        f"HTF {smc['htf_bias']} | Zone-first (OTE bonus) | Max SL ${MAX_SL_USD}"
    )
    if zone_quality:
        rationale += f" | {zone_quality}"
    if fund.get("risk_level") in ("HIGH", "MEDIUM"):
        rationale += f" | ⚠ NEWS: {fund.get('next_high_impact')} ({fund.get('minutes_until_next')}m)"

    return {
        "symbol": decoded, "timeframe": timeframe,
        "market_structure": smc["bos_choch"],
        "technical_layer": {
            "order_block": smc["order_block"], "fair_value_gap": smc["fvg"],
            "liquidity_pool": smc["liquidity"], "institutional_bias": smc["bias"],
            "htf_bias": smc["htf_bias"], "smc_score": smc["score"],
            "score_label": smc["score_label"], "score_detail": smc["score_detail"],
            "entry_reason": smc["entry_reason"], "confluence_quality": smc["confluence_quality"]
        },
        "fundamental_layer": fund,
        "master_decision": {
            "action": action,
            "confidence_score": f"{confidence}% ({smc['score_label']})",
            "execution_status": status
        },
        "execution_parameters": {
            "entry_price": str(entry), "stop_loss": str(sl),
            "take_profit_1": str(tp1), "take_profit_2": str(tp2),
            "risk_to_reward_ratio": rrr, "atr_used": atr_val,
            "current_price": str(round(current_price, 2)),
            "price_source": price_data.get("source", "biquote"),
            "last_candle_time": last_candle_time,
            "entry_reason": smc["entry_reason"],
            "zone_quality": zone_quality or "-"
        },
        "learning": {
            "score_adjust": learning_stats.get("score_adjust", 0),
            "min_score_now": min_sc,
            "tp_hits": learning_stats.get("tp_hits", 0),
            "sl_hits": learning_stats.get("sl_hits", 0),
            "miss_entries": learning_stats.get("miss_entries", 0),
            "total_closed": learning_stats.get("total_closed", 0),
            "learn_note": smc.get("learn_note", "neutral"),
            "pattern_key": smc.get("pattern_key", "")
        },
        "ai_rationale": rationale
    }

@app.get("/api/learning-stats")
async def learning_stats_endpoint():
    total = learning_stats.get("tp_hits", 0) + learning_stats.get("sl_hits", 0)
    wr = round(learning_stats["tp_hits"] / total * 100, 1) if total else 0
    return {
        "winrate_percent": wr,
        "tp_hits": learning_stats.get("tp_hits", 0),
        "sl_hits": learning_stats.get("sl_hits", 0),
        "miss_entries": learning_stats.get("miss_entries", 0),
        "score_adjust": learning_stats.get("score_adjust", 0),
        "min_score_now": get_dynamic_min_score(),
        "recent_results": learning_stats.get("recent_results", [])[:10],
        "last_update": learning_stats.get("last_update")
    }

@app.get("/api/fundamental/{symbol}")
async def get_fund(symbol: str = "XAUUSD"):
    return {"symbol": norm_symbol(symbol), "analysis": analyze_fundamental_xau()}

@app.get("/api/trade-history")
async def get_history():
    return {"history": trade_history}

@app.get("/api/reset-history")
async def reset_history():
    global trade_history
    trade_history = []
    save_history()
    return {"status": "success"}

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": "12.5",
        "data_source": "biquote.io",
        "max_sl_usd": MAX_SL_USD,
        "entry_mode": "zone_first_ote_bonus",
        "miss_entry_fixed": True,
        "news_blocks_signal": False,
        "trades": len(trade_history),
        "min_score": get_dynamic_min_score()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

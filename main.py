# main.py - Apex SMC Intelligence v10.1
# Fitur: MISS ENTRY + Alasan Entry (OB/FVG/BOS) + Pair jelas di History

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import time, os, json, requests, numpy as np
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SMC-Engine")

app = FastAPI(title="Apex SMC Intelligence", version="10.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "ed06e6d76c6e42d88fdf510856d9b900")
HISTORY_FILE = "trade_history.json"
CACHE_TTL = 8
FUND_CACHE_TTL = 300

cache_store = {}
price_cache = {}
fundamental_cache = {"data": None, "time": 0}
trade_history: List[Dict] = []

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
            json.dump(trade_history[:100], f, indent=2)
    except Exception as e:
        logger.error(e)

load_history()

# ==================== REAL-TIME PRICE (biquote) ====================
def fetch_realtime_price(symbol: str) -> Dict:
    key = f"price_{symbol}"
    now = time.time()
    if key in price_cache and (now - price_cache[key]["time"]) < 3:
        return price_cache[key]["data"]

    try:
        clean = symbol.replace("/", "").upper()
        r = requests.get(f"https://biquote.io/api/{clean}", timeout=5)
        data = r.json()
        if "mid" in data:
            result = {
                "symbol": symbol,
                "price": float(data["mid"]),
                "bid": float(data.get("bid", data["mid"])),
                "ask": float(data.get("ask", data["mid"])),
                "spread": float(data.get("spread", 0)),
                "timestamp": data.get("timestamp") or data.get("lastQuoteAt"),
                "source": data.get("source", "biquote-MT5"),
                "stale": data.get("stale", False)
            }
            price_cache[key] = {"time": now, "data": result}
            return result
    except Exception as e:
        logger.error(f"biquote error: {e}")

    candles = fetch_candles(symbol, "1min", 5)
    if candles:
        result = {
            "symbol": symbol,
            "price": candles[-1]["close"],
            "bid": candles[-1]["close"],
            "ask": candles[-1]["close"],
            "spread": 0,
            "timestamp": candles[-1]["datetime"],
            "source": "candle_fallback",
            "stale": True
        }
        price_cache[key] = {"time": now, "data": result}
        return result
    return {"symbol": symbol, "price": 0, "source": "error", "stale": True}

# ==================== CANDLES ====================
def fetch_candles(symbol: str, interval: str = "5min", outputsize: int = 100, force: bool = False) -> List[Dict]:
    key = f"{symbol}_{interval}"
    now = time.time()
    if not force and key in cache_store and (now - cache_store[key]["time"]) < CACHE_TTL:
        return cache_store[key]["data"]

    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_DATA_API_KEY}"
        r = requests.get(url, timeout=8)
        data = r.json()
        if "values" not in data:
            return cache_store.get(key, {}).get("data", [])
        candles = [{
            "datetime": c["datetime"],
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "volume": float(c.get("volume") or 0)
        } for c in reversed(data["values"])]
        cache_store[key] = {"time": now, "data": candles}
        return candles
    except Exception as e:
        logger.error(e)
        return cache_store.get(key, {}).get("data", [])

def calculate_atr(candles, period=14):
    if len(candles) < period + 1: return 1.5
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return round(float(np.mean(trs[-period:])), 5)

def find_swings(highs, lows, left=2, right=2):
    sh, sl = [], []
    for i in range(left, len(highs) - right):
        if highs[i] == max(highs[i-left:i+right+1]): sh.append((i, highs[i]))
        if lows[i] == min(lows[i-left:i+right+1]): sl.append((i, lows[i]))
    return sh, sl

# ==================== SMC ANALYSIS ====================
def analyze_pure_smc(candles):
    if len(candles) < 30:
        return {
            "bias": "NEUTRAL", "bos_choch": "Insufficient Data", "order_block": "None",
            "fvg": "None", "liquidity": "Consolidating", "score": 40, "setup": "WAIT",
            "current_price": candles[-1]["close"] if candles else 0, "atr": 1.5,
            "last_candle_time": candles[-1]["datetime"] if candles else None,
            "entry_reason": "Insufficient data"
        }

    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    current_price = float(closes[-1])
    atr = calculate_atr(candles)
    last_time = candles[-1]["datetime"]

    sh, sl = find_swings(highs, lows)
    structure, bias = "RANGE / CHOPPY", "NEUTRAL"

    if len(sh) >= 2 and len(sl) >= 2:
        if sh[-1][1] > sh[-2][1] and sl[-1][1] > sl[-2][1]:
            structure, bias = "BULLISH BOS (HH+HL)", "BULLISH"
        elif sh[-1][1] < sh[-2][1] and sl[-1][1] < sl[-2][1]:
            structure, bias = "BEARISH BOS (LH+LL)", "BEARISH"
        elif sh[-1][1] > sh[-2][1] and sl[-1][1] < sl[-2][1]:
            structure, bias = "CHoCH → Bullish", "BULLISH"
        elif sh[-1][1] < sh[-2][1] and sl[-1][1] > sl[-2][1]:
            structure, bias = "CHoCH → Bearish", "BEARISH"

    rh, rl = np.max(highs[-12:-2]), np.min(lows[-12:-2])
    if current_price > rh and bias == "NEUTRAL":
        structure, bias = "BULLISH BOS", "BULLISH"
    elif current_price < rl and bias == "NEUTRAL":
        structure, bias = "BEARISH BOS", "BEARISH"

    liq = "Protected"
    if highs[-1] > rh and current_price < rh * 0.998:
        liq = "BSL Swept"
        if bias != "BULLISH": bias = "BEARISH"
    elif lows[-1] < rl and current_price > rl * 1.002:
        liq = "SSL Swept"
        if bias != "BEARISH": bias = "BULLISH"

    fvg = "No Valid FVG"
    for i in range(len(candles)-3, max(len(candles)-8, 2), -1):
        if candles[i]["low"] > candles[i-2]["high"]:
            fvg = f"Bullish FVG [{candles[i-2]['high']:.5f}-{candles[i]['low']:.5f}]"
            break
        if candles[i]["high"] < candles[i-2]["low"]:
            fvg = f"Bearish FVG [{candles[i]['high']:.5f}-{candles[i-2]['low']:.5f}]"
            break

    ob = "None"
    for i in range(len(candles)-3, max(len(candles)-20, 2), -1):
        if bias == "BULLISH" and candles[i]["close"] < candles[i]["open"]:
            if candles[i+1]["close"] > candles[i]["high"]:
                ob = f"Bullish OB @ {candles[i]['low']:.5f}-{candles[i]['high']:.5f}"
                break
        if bias == "BEARISH" and candles[i]["close"] > candles[i]["open"]:
            if candles[i+1]["close"] < candles[i]["low"]:
                ob = f"Bearish OB @ {candles[i]['low']:.5f}-{candles[i]['high']:.5f}"
                break

    # ===== ENTRY REASON (Alasan Valid) =====
    reasons = []
    if "BOS" in structure or "CHoCH" in structure:
        reasons.append(structure)
    if "OB" in ob and ob != "None":
        reasons.append(ob)
    if "FVG" in fvg and "No Valid" not in fvg:
        reasons.append(fvg)
    if "Swept" in liq:
        reasons.append(liq)

    entry_reason = " + ".join(reasons) if reasons else "No clear confluence"

    score = 55
    if "BOS" in structure: score += 18
    if "CHoCH" in structure: score += 12
    if "Swept" in liq: score += 15
    if "FVG" in fvg and "No Valid" not in fvg: score += 10
    if "OB" in ob and ob != "None": score += 8
    score = min(score, 96)

    setup = "WAIT FOR MITIGATION"
    if bias == "BULLISH" and score >= 70 and len(reasons) >= 1:
        setup = "BUY LIMIT (OTE)"
    elif bias == "BEARISH" and score >= 70 and len(reasons) >= 1:
        setup = "SELL LIMIT (OTE)"

    return {
        "bias": bias,
        "bos_choch": structure,
        "order_block": ob,
        "fvg": fvg,
        "liquidity": liq,
        "score": score,
        "setup": setup,
        "current_price": current_price,
        "atr": atr,
        "last_candle_time": last_time,
        "entry_reason": entry_reason
    }

# ==================== FUNDAMENTAL ====================
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
    except: pass
    return fundamental_cache["data"] or []

def analyze_fundamental_xau():
    events = fetch_high_impact()
    now = datetime.now(timezone.utc)
    critical = ["nonfarm", "nfp", "payroll", "cpi", "core cpi", "pce", "fomc", "interest rate", "rate decision", "fed", "adp", "gdp"]
    next_ev, min_m, upcoming = None, 99999, []

    for ev in events:
        try:
            et = datetime.fromisoformat(ev["time"].replace("Z", "+00:00"))
            mins = (et - now).total_seconds() / 60
            if mins < -45: continue
            name = ev["name"]
            if any(k in name.lower() for k in critical) or ev.get("importance") == "high":
                info = {"name": name, "minutes_left": round(mins, 1)}
                upcoming.append(info)
                if 0 <= mins < min_m:
                    min_m, next_ev = mins, info
        except: continue

    if not next_ev:
        return {"scalping_status": "SAFE FOR SCALPING", "risk_level": "LOW",
                "recommendation": "Tidak ada High Impact. Scalping aman.",
                "next_high_impact": "None", "minutes_until_next": None, "upcoming_events": []}
    if next_ev["minutes_left"] <= 90:
        return {"scalping_status": "DANGER - HIGH IMPACT SOON", "risk_level": "EXTREME",
                "recommendation": f"HINDARI SCALPING! {next_ev['name']} dalam {next_ev['minutes_left']} menit.",
                "next_high_impact": next_ev["name"], "minutes_until_next": next_ev["minutes_left"],
                "upcoming_events": upcoming[:4]}
    if next_ev["minutes_left"] <= 240:
        return {"scalping_status": "CAUTION - HIGH IMPACT COMING", "risk_level": "HIGH",
                "recommendation": f"Hati-hati. {next_ev['name']} dalam {round(next_ev['minutes_left']/60, 1)} jam.",
                "next_high_impact": next_ev["name"], "minutes_until_next": next_ev["minutes_left"],
                "upcoming_events": upcoming[:4]}
    return {"scalping_status": "SAFE FOR SCALPING", "risk_level": "LOW",
            "recommendation": f"Aman. Next event masih {round(next_ev['minutes_left']/60, 1)} jam lagi.",
            "next_high_impact": next_ev["name"], "minutes_until_next": next_ev["minutes_left"],
            "upcoming_events": upcoming[:4]}

# ==================== TRADE MANAGEMENT (dengan MISS ENTRY) ====================
def manage_trades(symbol: str, current_price: float):
    global trade_history
    changed = False

    for t in trade_history:
        if t["symbol"] != symbol:
            continue

        entry = float(t["entry"])
        sl = float(t["sl"])
        tp1 = float(t["tp1"])
        is_buy = "BUY" in t["type"]

        # 1. PENDING ENTRY
        if t["status"] == "PENDING ENTRY":
            filled = False
            missed = False

            if is_buy:
                if current_price <= entry:
                    filled = True
                elif current_price >= tp1:
                    missed = True
            else:
                if current_price >= entry:
                    filled = True
                elif current_price <= tp1:
                    missed = True

            if filled:
                t["status"] = "FILLED & ACTIVE"
                t["fill_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                t["fill_price"] = str(round(current_price, 5))
                changed = True
                logger.info(f"Trade #{t['id']} FILLED")
            elif missed:
                t["status"] = "CLOSED - MISS ENTRY"
                t["close_price"] = str(round(current_price, 5))
                t["close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                logger.info(f"Trade #{t['id']} MISS ENTRY")

        # 2. FILLED & ACTIVE
        elif t["status"] == "FILLED & ACTIVE":
            hit = None
            if is_buy:
                if current_price <= sl: hit = "SL HIT"
                elif current_price >= tp1: hit = "TP1 HIT"
            else:
                if current_price >= sl: hit = "SL HIT"
                elif current_price <= tp1: hit = "TP1 HIT"

            if hit:
                t["status"] = f"CLOSED - {hit}"
                t["close_price"] = str(round(current_price, 5))
                t["close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                logger.info(f"Trade #{t['id']} {hit}")

    if changed:
        save_history()
    return changed

# ==================== ENDPOINTS ====================
@app.get("/api/price/{symbol:path}")
async def get_price(symbol: str):
    decoded = symbol.replace("'", "").strip().upper()
    return fetch_realtime_price(decoded)

@app.get("/api/max-intelligence-signal/{symbol:path}")
async def get_signal(
    symbol: str,
    timeframe: str = Query("5min"),
    custom_atr: Optional[float] = Query(None),
    auto_lock: bool = Query(True)
):
    decoded = symbol.replace("'", "").strip().upper()
    if not decoded:
        raise HTTPException(400, "Symbol required")

    # Harga real-time
    price_data = fetch_realtime_price(decoded)
    current_price = price_data["price"]

    # Candles
    candles = fetch_candles(decoded, timeframe, force=True)
    if not candles:
        raise HTTPException(503, "Cannot fetch market data")

    if current_price > 0:
        candles[-1]["close"] = current_price

    last_candle_time = candles[-1]["datetime"]
    manage_trades(decoded, current_price)

    # Cek trade aktif / pending
    active = next(
        (t for t in trade_history if t["symbol"] == decoded and t["status"] in ["PENDING ENTRY", "FILLED & ACTIVE"]),
        None
    )

    if active:
        fund = analyze_fundamental_xau()
        return {
            "symbol": decoded,
            "timeframe": timeframe,
            "market_structure": f"Trade Active: {active['status']}",
            "technical_layer": {
                "order_block": active.get("order_block", "-"),
                "fair_value_gap": "-",
                "liquidity_pool": "-",
                "institutional_bias": "HOLD" if active["status"] == "FILLED & ACTIVE" else "WAITING ENTRY",
                "entry_reason": active.get("entry_reason", "-")
            },
            "fundamental_layer": fund,
            "master_decision": {
                "action": f"{'HOLD' if active['status'] == 'FILLED & ACTIVE' else 'WAITING'} ({active['type']})",
                "confidence_score": "100%",
                "execution_status": active["status"]
            },
            "execution_parameters": {
                "entry_price": active["entry"],
                "stop_loss": active["sl"],
                "take_profit_1": active["tp1"],
                "take_profit_2": active["tp2"],
                "risk_to_reward_ratio": active["rrr"],
                "atr_used": active.get("atr_used", "-"),
                "current_price": str(round(current_price, 5)),
                "price_source": price_data.get("source", "-"),
                "last_candle_time": last_candle_time,
                "entry_reason": active.get("entry_reason", "-")
            },
            "ai_rationale": f"Pair: {decoded} | Status: {active['status']} | Reason: {active.get('entry_reason', '-')}"
        }

    # Analisa baru
    smc = analyze_pure_smc(candles)
    fund = analyze_fundamental_xau()
    atr_val = custom_atr if custom_atr and custom_atr > 0 else smc["atr"]

    action = smc["setup"]
    confidence = smc["score"]
    status = "IDLE"
    entry = sl = tp1 = tp2 = "-"
    rrr = "0.0"

    if fund["risk_level"] in ["EXTREME", "HIGH"]:
        action = "WAIT - HIGH IMPACT NEWS RISK"
        status = "BLOCKED BY FUNDAMENTAL"
        confidence = max(25, confidence - 35)
    else:
        if "BUY" in action:
            entry = round(current_price - (atr_val * 0.35), 5)
            sl = round(entry - (atr_val * 2.0), 5)
            tp1 = round(entry + (atr_val * 3.2), 5)
            tp2 = round(entry + (atr_val * 5.5), 5)
            rrr = "1:2.7"
            status = "PENDING ENTRY" if auto_lock else "SIGNAL READY"
        elif "SELL" in action:
            entry = round(current_price + (atr_val * 0.35), 5)
            sl = round(entry + (atr_val * 2.0), 5)
            tp1 = round(entry - (atr_val * 3.2), 5)
            tp2 = round(entry - (atr_val * 5.5), 5)
            rrr = "1:2.7"
            status = "PENDING ENTRY" if auto_lock else "SIGNAL READY"

    # Simpan trade + alasan entry + pair
    if status == "PENDING ENTRY":
        new_trade = {
            "id": len(trade_history) + 1,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": decoded,                    # ← Pair jelas
            "type": action,
            "entry": str(entry),
            "sl": str(sl),
            "tp1": str(tp1),
            "tp2": str(tp2),
            "rrr": rrr,
            "status": status,
            "atr_used": atr_val,
            "order_block": smc["order_block"],
            "entry_reason": smc["entry_reason"],  # ← Alasan valid (OB/FVG/BOS)
            "bias": smc["bias"]
        }
        trade_history.insert(0, new_trade)
        save_history()

    return {
        "symbol": decoded,
        "timeframe": timeframe,
        "market_structure": smc["bos_choch"],
        "technical_layer": {
            "order_block": smc["order_block"],
            "fair_value_gap": smc["fvg"],
            "liquidity_pool": smc["liquidity"],
            "institutional_bias": smc["bias"],
            "smc_score": smc["score"],
            "entry_reason": smc["entry_reason"]
        },
        "fundamental_layer": fund,
        "master_decision": {
            "action": action,
            "confidence_score": f"{confidence}%",
            "execution_status": status
        },
        "execution_parameters": {
            "entry_price": str(entry),
            "stop_loss": str(sl),
            "take_profit_1": str(tp1),
            "take_profit_2": str(tp2),
            "risk_to_reward_ratio": rrr,
            "atr_used": atr_val,
            "current_price": str(round(current_price, 5)),
            "price_source": price_data.get("source", "-"),
            "last_candle_time": last_candle_time,
            "entry_reason": smc["entry_reason"]
        },
        "ai_rationale": f"Pair: {decoded} | {smc['entry_reason']} | Fund: {fund['scalping_status']}"
    }

@app.get("/api/fundamental/{symbol}")
async def get_fund(symbol: str = "XAUUSD"):
    return {"symbol": symbol.upper(), "analysis": analyze_fundamental_xau()}

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
    return {"status": "healthy", "version": "10.1", "trades": len(trade_history)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
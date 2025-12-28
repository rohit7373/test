import os, time, threading, uuid, math
import pandas as pd
import pandas_ta as ta
import ccxt
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uvicorn
from pymongo import MongoClient
from bson import ObjectId

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
exchange = ccxt.binance({"enableRateLimit": True})
SYMBOL = "BTC/USDT"
MONGO_URL = os.environ.get("MONGO_URL")
VALID_TFS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"]

# --- DATABASE HANDLES ---
db = None
state_collection = None
trades_collection = None
history_collection = None

# --- STATE ---
STATE = {
    "running": False,
    "wallet": 1000.0,
    "unrealized": 0.0, 
    "openTrades": [],  
    "history": []     
}

# Default configuration (will be overwritten by DB on startup)
BOT_CONFIG = {
    "stf1": "5m", "stf2": "15m",
    "ltf1": "1h", "ltf2": "4h",
    "stf_logic": ">", 
    "ltf_logic": ">", 
    "entry_mode": "BOTH", 
    "qty": 0.01,
    "fee": 0.1,  # Fee rate in percent
    "tp": None,  
    "sl": None,  
}

# Caching for market data to reduce CCXT API calls
CACHE = {"last_update": 0, "data": None}

# --- Pydantic Models for Requests ---
class BotStartReq(BaseModel):
    stf1: str; stf2: str; ltf1: str; ltf2: str
    stf_logic: str; ltf_logic: str
    entry_mode: str
    qty: float; fee: float
    sl: Optional[float] = None; tp: Optional[float] = None

class ManualOrder(BaseModel):
    side: str; qty: float; type: str; sl: Optional[float] = None; tp: Optional[float] = None

class CloseTradeReq(BaseModel):
    id: str

# --- UTILITIES ---

def clean_data(data):
    """Aggressively cleans data to ensure valid JSON response."""
    if data is None: return None
    if isinstance(data, list): return [clean_data(x) for x in data]
    if isinstance(data, dict):
        return {k: (str(v) if k == "_id" else clean_data(v)) for k, v in data.items()}
    if isinstance(data, ObjectId): return str(data)
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data): return None
        return data
    if isinstance(data, (np.integer, np.int64, np.int32)): return int(data)
    if isinstance(data, (np.floating, np.float64, np.float32)):
        if np.isnan(data) or np.isinf(data): return None
        return float(data)
    if isinstance(data, np.ndarray): return clean_data(data.tolist())
    if isinstance(data, datetime): return data.isoformat()
    return data

def update_db():
    """Saves wallet, running state, and BOT_CONFIG to MongoDB."""
    if state_collection is not None:
        try:
            state_collection.update_one(
                {"_id": "global_state"},
                {"$set": {"wallet": STATE["wallet"], "running": STATE["running"], "config": BOT_CONFIG}},
                upsert=True
            )
        except Exception as e:
            print(f"DB Update Failed: {e}")

def fetch_candles(tf_key, limit=300):
    """Fetch and prepare OHLCV data with RSI, returning a DataFrame."""
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, tf_key, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = df['time'] // 1000  # Convert milliseconds to seconds
        df.ta.rsi(length=14, append=True)
        df.rename(columns={'RSI_14': 'rsi'}, inplace=True)
        df = df[['time', 'open', 'high', 'low', 'close', 'rsi']].dropna()
        return df
    except Exception as e:
        return pd.DataFrame()

def last_price():
    """Fetches the current price of the trading symbol."""
    try: return float(exchange.fetch_ticker(SYMBOL)["last"])
    except: return 0.0

def pack_df_to_list(df): 
    """Converts a DataFrame to a list of dicts for FastAPI response."""
    if df.empty: return [] 
    return df[["time","open","high","low","close","rsi"]].to_dict("records")

def calculate_pnl():
    """Calculates total unrealized PnL for all open trades, including fees."""
    pnl = 0.0
    current_price = last_price()
    if not current_price: return 0.0
    
    for t in STATE["openTrades"]:
        entry = t['entryPrice']
        size = t['size']
        fee = t.get('fee_rate', BOT_CONFIG['fee']) / 100.0
        
        # Estimate total fee cost (entry + exit)
        total_fee_cost = (entry * size * fee) + (current_price * size * fee)
        
        if t['side'] == 'LONG':
            unrealized_pnl = (current_price - entry) * size - total_fee_cost
        else: # SHORT
            unrealized_pnl = (entry - current_price) * size - total_fee_cost

        t['pnl'] = unrealized_pnl
        pnl += unrealized_pnl
        
    STATE["unrealized"] = pnl
    return pnl

def check_condition(val1, op, val2):
    if op == ">": return val1 > val2
    if op == "<": return val1 < val2
    return False

def check_cross(prev1, curr1, op, prev2, curr2):
    if op == ">":
        return (prev1 <= prev2) and (curr1 > curr2)
    if op == "<":
        return (prev1 >= prev2) and (curr1 < curr2)
    return False

def open_trade(side, price, logic="Manual"):
    """Opens a new trade and saves it to state and DB."""
    t = {
        "id": str(uuid.uuid4())[:8], "side": side, "size": float(BOT_CONFIG["qty"]),
        "entryPrice": float(price), "sl": BOT_CONFIG["sl"], "tp": BOT_CONFIG["tp"],
        "fee_rate": float(BOT_CONFIG["fee"]), "pnl": 0.0, "auto": True, "logic": logic,
        "time": datetime.now().isoformat() 
    }
    STATE["openTrades"].append(t)
    
    if trades_collection is not None: 
        try: trades_collection.insert_one(t.copy())
        except: pass
    update_db()

def close_trade(t, price, reason):
    """Closes an existing trade, calculates realized PnL, and updates history."""
    if t in STATE["openTrades"]:
        
        entry = t['entryPrice']
        size = t['size']
        fee = t.get('fee_rate', BOT_CONFIG['fee']) / 100.0

        # Total fee calculation (entry fee + exit fee)
        total_fee_cost = (entry * size * fee) + (price * size * fee)
        
        if t['side'] == 'LONG':
            realized_pnl = (price - entry) * size - total_fee_cost
        else: # SHORT
            realized_pnl = (entry - price) * size - total_fee_cost

        STATE["wallet"] += realized_pnl
        
        # History record
        h = {
            "time": datetime.now().isoformat(), "side": t["side"],
            "price": float(price), "qty": t["size"],
            "realizedPnl": realized_pnl, "logic": t.get("logic", "N/A"), "reason": reason
        }
        STATE["history"].append(h)
        
        # Remove from openTrades
        STATE["openTrades"].remove(t)
        
        # DB Updates
        if trades_collection is not None: 
            try: trades_collection.delete_one({"id": t["id"]})
            except: pass
        if history_collection is not None:
            try: history_collection.insert_one(h.copy())
            except: pass
        update_db()

def check_entry_signal(df_stf_f, df_stf_s, df_ltf_f, df_ltf_s):
    """Checks for the entry signal based on the BOT_CONFIG entry_mode."""
    min_len = 2
    if any(len(df) < min_len for df in [df_stf_f, df_stf_s, df_ltf_f, df_ltf_s]):
        return None, None

    # Get current and previous RSI values
    stf_f, stf_s = df_stf_f["rsi"].iloc[-1], df_stf_s["rsi"].iloc[-1]
    stf_f_prev, stf_s_prev = df_stf_f["rsi"].iloc[-2], df_stf_s["rsi"].iloc[-2]
    ltf_f, ltf_s = df_ltf_f["rsi"].iloc[-1], df_ltf_s["rsi"].iloc[-1]
    ltf_f_prev, ltf_s_prev = df_ltf_f["rsi"].iloc[-2], df_ltf_s["rsi"].iloc[-2]
    
    if math.isnan(stf_f) or math.isnan(ltf_f): return None, None
    
    stf_op = BOT_CONFIG["stf_logic"]
    ltf_op = BOT_CONFIG["ltf_logic"]
    entry_mode = BOT_CONFIG["entry_mode"]
    
    # --- Evaluate Conditions ---
    
    # 1. STF Entry Cross Check
    stf_long_cross = check_cross(stf_f_prev, stf_f, '>', stf_s_prev, stf_s)
    stf_short_cross = check_cross(stf_f_prev, stf_f, '<', stf_s_prev, stf_s)

    # 2. LTF Filter Check (LTF is in the direction of the trade)
    ltf_filter_bullish = check_condition(ltf_f, ltf_op, ltf_s)
    ltf_filter_bearish = check_condition(ltf_s, ltf_op, ltf_f) 
    
    # 3. LTF Cross Check (Used for LTF_ONLY mode)
    ltf_long_cross = check_cross(ltf_f_prev, ltf_f, '>', ltf_s_prev, ltf_s)
    ltf_short_cross = check_cross(ltf_f_prev, ltf_f, '<', ltf_s_prev, ltf_s)

    logic_str = ""
    trade_allowed = False
    side = None
    
    if entry_mode == "BOTH":
        # LONG: STF Cross UP AND LTF Filter is Bullish
        if stf_long_cross and ltf_filter_bullish:
            trade_allowed = True; side = "LONG"
            logic_str = f"STF Cross UP + LTF Filter ({ltf_op})"
        # SHORT: STF Cross DOWN AND LTF Filter is Bearish
        elif stf_short_cross and ltf_filter_bearish:
            trade_allowed = True; side = "SHORT"
            logic_str = f"STF Cross DOWN + LTF Filter ({ltf_op})"

    elif entry_mode == "STF_ONLY":
        if stf_long_cross:
            trade_allowed = True; side = "LONG"
            logic_str = "STF Cross UP ONLY"
        elif stf_short_cross:
            trade_allowed = True; side = "SHORT"
            logic_str = "STF Cross DOWN ONLY"

    elif entry_mode == "LTF_ONLY":
        if ltf_long_cross:
            trade_allowed = True; side = "LONG"
            logic_str = "LTF Cross UP ONLY"
        elif ltf_short_cross:
            trade_allowed = True; side = "SHORT"
            logic_str = "LTF Cross DOWN ONLY"

    return (side, logic_str) if trade_allowed else (None, None)

def check_exit_signal(current_price):
    """Checks for TP/SL signals based on current price for all open trades."""
    trades_to_close = []
    
    for t in STATE["openTrades"]:
        exit_reason = None
        
        if t['sl'] is not None and t['sl'] > 0:
            if t['side'] == 'LONG' and current_price <= t['sl']: exit_reason = "SL Hit"
            elif t['side'] == 'SHORT' and current_price >= t['sl']: exit_reason = "SL Hit"

        if exit_reason is None and t['tp'] is not None and t['tp'] > 0:
            if t['side'] == 'LONG' and current_price >= t['tp']: exit_reason = "TP Hit"
            elif t['side'] == 'SHORT' and current_price <= t['tp']: exit_reason = "TP Hit"
        
        if exit_reason:
            trades_to_close.append((t, exit_reason))

    for t, reason in trades_to_close:
        close_trade(t, current_price, reason)

# --- BOT LOOP ---
def bot_loop():
    while True:
        try:
            current_price = last_price()
            if not current_price:
                time.sleep(3)
                continue

            calculate_pnl()
            check_exit_signal(current_price)
            
            if STATE["running"]:
                
                # Bot should only trade if it has no auto-trades open
                if not any(t.get('auto') for t in STATE["openTrades"]):
                    
                    df_stf_f = fetch_candles(BOT_CONFIG['stf1'], limit=50)
                    df_stf_s = fetch_candles(BOT_CONFIG['stf2'], limit=50)
                    df_ltf_f = fetch_candles(BOT_CONFIG['ltf1'], limit=50)
                    df_ltf_s = fetch_candles(BOT_CONFIG['ltf2'], limit=50)
                    
                    side, logic = check_entry_signal(df_stf_f, df_stf_s, df_ltf_f, df_ltf_s)

                    if side:
                        open_trade(side, current_price, logic)

            update_db() # Save State (wallet/running/config)
                
            time.sleep(5) # Poll every 5 seconds

        except Exception as e:
            print(f"Bot Loop Error: {e}")
            time.sleep(5)

# --- DB CONNECTION (Kept the same) ---

if not MONGO_URL:
    print("⚠️ MONGO_URL not found. Using In-Memory Mode.")
else:
    try:
        client = MongoClient(MONGO_URL)
        db = client.trading_bot 
        state_collection = db.state_collection
        trades_collection = db.open_trades
        history_collection = db.history
        print("✅ MongoDB Connected")
        
        # Load State
        saved = state_collection.find_one({"_id": "global_state"})
        if saved:
            STATE["wallet"] = saved.get("wallet", 1000.0)
            STATE["running"] = saved.get("running", False)
            if "config" in saved:
                merged_config = BOT_CONFIG.copy()
                for k,v in saved["config"].items():
                    if k in merged_config: merged_config[k] = v
                BOT_CONFIG = merged_config
        else:
            state_collection.insert_one({"_id": "global_state", "wallet": 1000.0, "running": False, "config": BOT_CONFIG})

        # Load Trades
        STATE["openTrades"] = clean_data(list(trades_collection.find({})))
        STATE["history"] = clean_data(list(history_collection.find().sort("time", -1).limit(100)))

    except Exception as e:
        print(f"❌ DB Error: {e}")

threading.Thread(target=bot_loop, daemon=True).start()

# --- API ENDPOINTS ---

@app.get("/api/config")
async def get_initial_config():
    """Serves the current BOT_CONFIG once for initial UI population."""
    return JSONResponse(content=clean_data(BOT_CONFIG))
    
@app.get("/api/chart_data/{timeframe}")
def chart_data(timeframe: str):
    """
    NEW ENDPOINT: Fetches candlestick and RSI data for a single requested timeframe 
    for dynamic chart display.
    """
    if timeframe not in VALID_TFS:
        raise HTTPException(status_code=400, detail="Invalid timeframe requested")
            
    df = fetch_candles(timeframe, limit=300)
    return JSONResponse(content=clean_data({
        "timeframe": timeframe,
        "data": pack_df_to_list(df)
    }))

@app.get("/api/market")
def market():
    """
    FIX: Only sends volatile data (price, state, trades, history). 
    Chart data is now fetched separately.
    """
    try:
        # Simple caching for high-frequency data
        if time.time() - CACHE["last_update"] < 3 and CACHE["data"] is not None:
            cached_data = CACHE["data"].copy()
            cached_data["price"] = last_price()
            cached_data["state"] = clean_data({"running": STATE["running"], "wallet": STATE["wallet"], "unrealized": STATE["unrealized"]})
            cached_data["openTrades"] = clean_data(STATE["openTrades"])
            cached_data["history"] = clean_data(STATE["history"])
            return JSONResponse(content=cached_data)
        
        data = {
            "price": last_price(),
            "state": {
                "running": STATE["running"],
                "wallet": STATE["wallet"],
                "unrealized": STATE["unrealized"],
            },
            "openTrades": STATE["openTrades"],
            "history": STATE["history"]
        }
        
        clean_response = clean_data(data)
        CACHE["data"] = clean_response
        CACHE["last_update"] = time.time()
        return JSONResponse(content=clean_response)

    except Exception as e:
        print(f"Market Endpoint Error: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/start")
def start(req: BotStartReq):
    try:
        if STATE["running"]:
            raise HTTPException(status_code=400, detail="Bot is already running")
            
        if req.stf_logic not in (">", "<") or req.ltf_logic not in (">", "<"):
             raise HTTPException(status_code=400, detail="Logic operators must be '>' or '<'")
        if req.entry_mode not in ("STF_ONLY", "LTF_ONLY", "BOTH"):
             raise HTTPException(status_code=400, detail="Entry mode must be 'STF_ONLY', 'LTF_ONLY', or 'BOTH'")
             
        BOT_CONFIG.update(req.dict())
        STATE["running"] = True
        update_db()
        return clean_data({"status": "started", "config": BOT_CONFIG})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/stop")
def stop():
    try:
        STATE["running"] = False
        update_db()
        return {"status": "stopped"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/manual/order")
def manual_order(order: ManualOrder):
    try:
        price = last_price()
        if not price:
             raise HTTPException(status_code=500, detail="Could not fetch current price.")
             
        t = {
            "id": str(uuid.uuid4())[:8], "side": order.side, "size": float(order.qty),
            "entryPrice": price, "sl": order.sl, "tp": order.tp, "fee_rate": BOT_CONFIG["fee"],
            "pnl": 0.0, "logic": "Manual", "time": datetime.now().isoformat()
        }
        STATE["openTrades"].append(t)
        
        if trades_collection is not None: 
            try: trades_collection.insert_one(t.copy())
            except: pass
            
        return clean_data({"status": "success", "trade": t})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/manual/close")
def manual_close(req: CloseTradeReq):
    try:
        t = next((x for x in STATE["openTrades"] if x["id"] == req.id), None)
        if t:
            close_trade(t, last_price(), "Manual Close")
            return {"status": "success"}
        raise HTTPException(status_code=404, detail="Trade not found")
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/manual/close-all")
def close_all():
    try:
        p = last_price()
        count = len(STATE["openTrades"])
        for t in list(STATE["openTrades"]): close_trade(t, p, "Close All")
        return {"status": "success", "closed": count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))

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

# Added new logic and entry keys, changed default TFs to be general
BOT_CONFIG = {
    "stf1": "5m", "stf2": "15m",
    "ltf1": "1h", "ltf2": "4h",
    "stf_logic": ">",  # New: Operator for stf1 vs stf2 cross
    "ltf_logic": ">",  # New: Operator for ltf1 vs ltf2 filter
    "entry_mode": "BOTH", # New: Options: "STF_ONLY", "LTF_ONLY", "BOTH"
    "tp": None,    "sl": None,
    "fee": 0.1,    "qty": 0.01
}

# --- BULLETPROOF DATA CLEANER ---
def clean_data(data):
    """
    Aggressively cleans data to ensure valid JSON response.
    Handles: MongoDB ObjectIds, Numpy types, NaNs, Infinity.
    """
    if data is None:
        return None
    
    # 1. Handle Lists
    if isinstance(data, list):
        return [clean_data(x) for x in data]
    
    # 2. Handle Dictionaries
    if isinstance(data, dict):
        return {k: (str(v) if k == "_id" else clean_data(v)) for k, v in data.items()}
    
    # 3. Handle MongoDB ObjectId
    if isinstance(data, ObjectId):
        return str(data)
    
    # 4. Handle Float (NaN / Infinity check)
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data

    # 5. Handle Numpy Types
    if isinstance(data, (np.integer, np.int64, np.int32)):
        return int(data)
    if isinstance(data, (np.floating, np.float64, np.float32)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    if isinstance(data, np.ndarray):
        return clean_data(data.tolist())

    return data

# --- DB CONNECTION ---
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
                # Merge existing config with new keys for backward compatibility
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

# --- HELPERS ---
# Expanded TF_MAP to include all options
TF_MAP = {"1m":"1m", "3m":"3m", "5m":"5m", "15m":"15m", "30m":"30m", "1h":"1h", "2h":"2h", "4h":"4h", "6h":"6h", "8h":"8h", "12h":"12h", "1d":"1d", "1w":"1w"}
CACHE = {"last_update": 0, "data": None}

def update_db():
    # FIX: Explicit check using 'is not None'
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
    try:
        tf = TF_MAP.get(tf_key, "1m")
        ohlcv = exchange.fetch_ohlcv(SYMBOL, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
        df["time"] = (df["time"] / 1000).astype(int)
        df["rsi"] = ta.rsi(df["close"], length=14)
        return df
    except:
        return pd.DataFrame()

def last_price():
    try: return float(exchange.fetch_ticker(SYMBOL)["last"])
    except: return 0.0

# --- CORE LOGIC HANDLER ---
def check_condition(val1, op, val2):
    if op == ">": return val1 > val2
    if op == "<": return val1 < val2
    return False

def check_cross(prev1, curr1, op, prev2, curr2):
    # Cross UP (for '>') or Cross DOWN (for '<')
    if op == ">":
        return (prev1 <= prev2) and (curr1 > curr2)
    if op == "<":
        return (prev1 >= prev2) and (curr1 < curr2)
    return False
    
# --- BOT LOOP ---
def bot_loop():
    while True:
        try:
            current_price = last_price()
            total_unrealized = 0.0
            trades_to_close = []
            
            # PnL Calculation and SL/TP check
            for t in STATE["openTrades"]:
                diff = current_price - t["entryPrice"]
                if t["side"] == "SHORT": diff = -diff
                
                fee = (float(t["size"]) * current_price) * (float(t.get("fee_rate", 0.1))/100)
                t["pnl"] = (diff * float(t["size"])) - fee
                total_unrealized += t["pnl"]
                
                sl, tp = t.get("sl"), t.get("tp")
                if t["side"] == "LONG":
                    if sl and current_price <= sl: trades_to_close.append(t)
                    if tp and current_price >= tp: trades_to_close.append(t)
                elif t["side"] == "SHORT":
                    if sl and current_price >= sl: trades_to_close.append(t)
                    if tp and current_price <= tp: trades_to_close.append(t)

            STATE["unrealized"] = total_unrealized
            for t in trades_to_close: close_trade(t, current_price, "SL/TP")

            # Auto Trading Logic
            if STATE["running"]:
                s1, s2 = BOT_CONFIG["stf1"], BOT_CONFIG["stf2"]
                l1, l2 = BOT_CONFIG["ltf1"], BOT_CONFIG["ltf2"]
                stf_op = BOT_CONFIG["stf_logic"]
                ltf_op = BOT_CONFIG["ltf_logic"]
                entry_mode = BOT_CONFIG["entry_mode"]
                
                # Fetch all required dataframes
                df_stf_f, df_stf_s = fetch_candles(s1), fetch_candles(s2)
                df_ltf_f, df_ltf_s = fetch_candles(l1), fetch_candles(l2)
                
                # Check for minimum data size
                min_len = 2
                if len(df_stf_f) < min_len or len(df_stf_s) < min_len or len(df_ltf_f) < min_len or len(df_ltf_s) < min_len:
                    time.sleep(2) # Wait if data is incomplete
                    continue

                try:
                    # Get current and previous RSI values
                    stf_f, stf_s = df_stf_f["rsi"].iloc[-1], df_stf_s["rsi"].iloc[-1]
                    stf_f_prev, stf_s_prev = df_stf_f["rsi"].iloc[-2], df_stf_s["rsi"].iloc[-2]
                    ltf_f, ltf_s = df_ltf_f["rsi"].iloc[-1], df_ltf_s["rsi"].iloc[-1]
                    
                    # Ensure current values are not NaN
                    if math.isnan(stf_f) or math.isnan(ltf_f): continue
                    
                    # --- Evaluate Conditions ---
                    
                    # LTF Condition (Filter/Confirmation): ltf1 [ltf_op] ltf2
                    ltf_condition = check_condition(ltf_f, ltf_op, ltf_s)
                    
                    # STF Condition (Entry): stf1 crosses stf2 according to stf_op
                    stf_condition = check_cross(stf_f_prev, stf_f, stf_op, stf_s_prev, stf_s)
                    
                    # Trade Direction: Based on the STF operator
                    side = "LONG" if stf_op == ">" else "SHORT"
                    
                    # Check if bot can open a trade (no auto trades open)
                    auto_open = not any(t.get('auto') for t in STATE["openTrades"])
                    
                    if auto_open:
                        trade_allowed = False
                        
                        if entry_mode == "BOTH":
                            trade_allowed = stf_condition and ltf_condition
                        elif entry_mode == "STF_ONLY":
                            trade_allowed = stf_condition
                        elif entry_mode == "LTF_ONLY":
                            # Use LTF cross as entry trigger
                            ltf_cross = check_cross(df_ltf_f["rsi"].iloc[-2], ltf_f, ltf_op, df_ltf_s["rsi"].iloc[-2], ltf_s)
                            trade_allowed = ltf_cross
                            # The side must match the LTF operator
                            side = "LONG" if ltf_op == ">" else "SHORT"

                        if trade_allowed:
                            logic_str = f"({s1} {stf_op} {s2}) & ({l1} {ltf_op} {l2}) / Mode: {entry_mode}"
                            open_trade(side, current_price, logic_str)
                            
                except Exception as e:
                    print(f"Logic Error: {e}") 
                    pass # Continue loop even on logic error

        except Exception as e:
            print(f"Loop Error: {e}")
        time.sleep(2)

# ... (open_trade and close_trade functions remain the same) ...

def open_trade(side, price, logic="Manual"):
    t = {
        "id": str(uuid.uuid4())[:8], "side": side, "size": float(BOT_CONFIG["qty"]),
        "entryPrice": float(price), "sl": BOT_CONFIG["sl"], "tp": BOT_CONFIG["tp"],
        "fee_rate": float(BOT_CONFIG["fee"]), "pnl": 0.0, "auto": True, "logic": logic,
        "time": datetime.now().isoformat()
    }
    STATE["openTrades"].append(t)
    
    # FIX: Explicit check
    if trades_collection is not None: 
        try: trades_collection.insert_one(t.copy())
        except: pass

def close_trade(t, price, reason):
    if t in STATE["openTrades"]:
        STATE["openTrades"].remove(t)
        h = {
            "time": datetime.now().isoformat(), "side": t["side"],
            "entryPrice": t["entryPrice"], "exitPrice": float(price), "price": float(price),
            "qty": t["size"], "realizedPnl": t["pnl"], "logic": t.get("logic"), "reason": reason
        }
        STATE["history"].append(h)
        STATE["wallet"] += t["pnl"]
        
        # FIX: Explicit checks
        if trades_collection is not None: 
            try: trades_collection.delete_one({"id": t["id"]})
            except: pass
        if history_collection is not None:
            try: history_collection.insert_one(h.copy())
            except: pass
        update_db()


threading.Thread(target=bot_loop, daemon=True).start()

# --- API MODELS ---
class BotStartReq(BaseModel):
    stf1: str; stf2: str; ltf1: str; ltf2: str
    stf_logic: str; ltf_logic: str # Added logic operators
    entry_mode: str # Added entry mode
    qty: float; fee: float = 0.0
    sl: Optional[float] = None; tp: Optional[float] = None

class ManualOrder(BaseModel):
    side: str; qty: float; type: str; sl: Optional[float] = None; tp: Optional[float] = None

class CloseTradeReq(BaseModel):
    id: str

# --- API ENDPOINTS ---

@app.get("/api/market")
def market():
    try:
        # Check if config needs to be updated from current BOT_CONFIG state
        # This ensures the frontend pulls the latest logic/mode config on refresh
        if CACHE["data"] and (time.time() - CACHE["last_update"] < 3): 
            # Update the config part of the cached data before returning
            CACHE["data"]["config"] = BOT_CONFIG
            CACHE["data"]["state"]["openTrades"] = STATE["openTrades"]
            CACHE["data"]["state"]["history"] = STATE["history"]
            return CACHE["data"]
        
        def pack(df): return [] if df.empty else df[["time","open","high","low","close","rsi"]].to_dict("records")
        
        data = {
            "price": last_price(),
            "stf1": pack(fetch_candles(BOT_CONFIG["stf1"])),
            "stf2": pack(fetch_candles(BOT_CONFIG["stf2"])),
            "ltf1": pack(fetch_candles(BOT_CONFIG["ltf1"])),
            "ltf2": pack(fetch_candles(BOT_CONFIG["ltf2"])),
            "config": BOT_CONFIG, # Send new config keys
            "state": STATE,
            "openTrades": STATE["openTrades"],
            "history": STATE["history"]
        }
        
        CACHE["data"] = clean_data(data)
        CACHE["last_update"] = time.time()
        return CACHE["data"]
    except Exception as e:
        print(f"Market Endpoint Error: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/start")
def start(req: BotStartReq):
    try:
        # Validate logic operators
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

# ... (stop, manual_order, manual_close, close_all remain the same) ...

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
        t = {
            "id": str(uuid.uuid4())[:8], "side": order.side, "size": float(order.qty),
            "entryPrice": price, "sl": order.sl, "tp": order.tp, "fee_rate": 0.1,
            "pnl": 0.0, "logic": "Manual", "time": datetime.now().isoformat()
        }
        STATE["openTrades"].append(t)
        
        # FIX: Explicit check
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

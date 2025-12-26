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

BOT_CONFIG = {
    "stf1": "1m", "stf2": "5m",
    "ltf1": "1h", "ltf2": "4h",
    "tp": None,   "sl": None,
    "fee": 0.1,   "qty": 0.01
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
                for k,v in saved["config"].items():
                    if k in BOT_CONFIG: BOT_CONFIG[k] = v
        else:
            state_collection.insert_one({"_id": "global_state", "wallet": 1000.0, "running": False, "config": BOT_CONFIG})

        # Load Trades
        STATE["openTrades"] = clean_data(list(trades_collection.find({})))
        STATE["history"] = clean_data(list(history_collection.find().sort("time", -1).limit(100)))

    except Exception as e:
        print(f"❌ DB Error: {e}")

# --- HELPERS ---
TF_MAP = {"1m":"1m", "3m":"3m", "5m":"5m", "15m":"15m", "1h":"1h", "2h":"2h", "4h":"4h", "1d":"1d", "1w":"1w"}
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

# --- BOT LOOP ---
def bot_loop():
    while True:
        try:
            current_price = last_price()
            total_unrealized = 0.0
            trades_to_close = []
            
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

            if STATE["running"]:
                s1, s2 = BOT_CONFIG["stf1"], BOT_CONFIG["stf2"]
                l1, l2 = BOT_CONFIG["ltf1"], BOT_CONFIG["ltf2"]
                df_stf_f, df_stf_s = fetch_candles(s1), fetch_candles(s2)
                df_ltf_f, df_ltf_s = fetch_candles(l1), fetch_candles(l2)
                
                if len(df_stf_f)>2 and len(df_stf_s)>2 and len(df_ltf_f)>2 and len(df_ltf_s)>2:
                    try:
                        stf_f, stf_s = df_stf_f["rsi"].iloc[-1], df_stf_s["rsi"].iloc[-1]
                        stf_f_prev, stf_s_prev = df_stf_f["rsi"].iloc[-2], df_stf_s["rsi"].iloc[-2]
                        ltf_f, ltf_s = df_ltf_f["rsi"].iloc[-1], df_ltf_s["rsi"].iloc[-1]
                        
                        if not math.isnan(stf_f) and not math.isnan(ltf_f):
                            auto_open = not any(t.get('auto') for t in STATE["openTrades"])
                            if auto_open:
                                if (ltf_f > ltf_s) and (stf_f_prev <= stf_s_prev and stf_f > stf_s):
                                    open_trade("LONG", current_price, f"{s1}>{s2} & {l1}>{l2}")
                                elif (ltf_f < ltf_s) and (stf_f_prev >= stf_s_prev and stf_f < stf_s):
                                    open_trade("SHORT", current_price, f"{s1}<{s2} & {l1}<{l2}")
                    except: pass
        except Exception as e:
            print(f"Loop Error: {e}")
        time.sleep(2)

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
        if CACHE["data"] and (time.time() - CACHE["last_update"] < 3): 
            return CACHE["data"]
        
        def pack(df): return [] if df.empty else df[["time","open","high","low","close","rsi"]].to_dict("records")
        
        data = {
            "price": last_price(),
            "stf1": pack(fetch_candles(BOT_CONFIG["stf1"])),
            "stf2": pack(fetch_candles(BOT_CONFIG["stf2"])),
            "ltf1": pack(fetch_candles(BOT_CONFIG["ltf1"])),
            "ltf2": pack(fetch_candles(BOT_CONFIG["ltf2"])),
            "config": BOT_CONFIG,
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

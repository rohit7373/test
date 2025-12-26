import os, time, threading, uuid
import pandas as pd
import pandas_ta as ta
import ccxt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uvicorn
from pymongo import MongoClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION & DB SETUP ---
exchange = ccxt.binance({"enableRateLimit": True})
SYMBOL = "BTC/USDT"
MONGO_URL = os.environ.get("MONGO_URL")

# Global DB Handles
db = None
state_collection = None
trades_collection = None
history_collection = None

# Default State
STATE = {
    "running": False,
    "wallet": 1000.0,
    "unrealized": 0.0, 
    "openTrades": [],  
    "history": []      
}

# Default Config
BOT_CONFIG = {
    "stf1": "1m", "stf2": "5m",
    "ltf1": "1h", "ltf2": "4h",
    "tp": None,   "sl": None,
    "fee": 0.1,   "qty": 0.01
}

# --- DATABASE CONNECTION & LOADING ---
if not MONGO_URL:
    print("❌ WARNING: MONGO_URL not found. Using in-memory mode (Data lost on restart).")
else:
    try:
        client = MongoClient(MONGO_URL)
        db = client.trading_bot 
        
        state_collection = db.state_collection
        trades_collection = db.open_trades
        history_collection = db.history
        
        print("✅ MongoDB Client Initialized and Connected")
        
        # 1. LOAD GLOBAL STATE (Wallet, Running Status, AND Config)
        saved_state = state_collection.find_one({"_id": "global_state"})
        if saved_state:
            STATE["wallet"] = saved_state.get("wallet", 1000.0)
            STATE["running"] = saved_state.get("running", False)
            
            # Load Saved Config if it exists
            if "config" in saved_state:
                saved_config = saved_state["config"]
                # Update BOT_CONFIG keys only if they exist in saved data
                for k in BOT_CONFIG:
                    if k in saved_config:
                        BOT_CONFIG[k] = saved_config[k]
        else:
            # Initialize DB if empty
            state_collection.insert_one({
                "_id": "global_state", 
                "wallet": 1000.0, 
                "running": False,
                "config": BOT_CONFIG
            })

        # 2. LOAD OPEN TRADES
        # This will automatically load 'sl', 'tp', 'fee_rate', 'logic' if they exist in DB
        db_trades = list(trades_collection.find({}))
        for t in db_trades:
            if "_id" in t: del t["_id"]
        STATE["openTrades"] = db_trades

        # 3. LOAD HISTORY
        db_history = list(history_collection.find().sort("time", -1).limit(100))
        for h in db_history:
            if "_id" in h: del h["_id"]
        STATE["history"] = db_history
        
        print(f"🔄 State Loaded: Wallet ${STATE['wallet']:.2f}, Trades: {len(STATE['openTrades'])}")

    except Exception as e:
        print(f"❌ FATAL: Could not connect to MongoDB: {e}")

TF_MAP = {
    "1m":"1m", "3m":"3m", "5m":"5m", "15m":"15m", "30m":"30m",
    "1h":"1h", "2h":"2h", "4h":"4h", "6h":"6h", "8h":"8h", "12h":"12h",
    "1d":"1d", "3d":"3d", "1w":"1w"
}

CACHE = {
    "last_update": 0,
    "data": None
}

# --- DATA MODELS ---
class ManualOrder(BaseModel):
    side: str
    qty: float
    type: str
    sl: Optional[float] = None 
    tp: Optional[float] = None

class BotStartReq(BaseModel):
    stf1: str
    stf2: str
    ltf1: str
    ltf2: str
    qty: float
    fee: float = 0.0
    sl: Optional[float] = None
    tp: Optional[float] = None

class CloseTradeReq(BaseModel):
    id: str

# --- HELPER FUNCTIONS ---
def update_db_state():
    """Updates wallet, running status, AND bot configuration in DB"""
    if state_collection:
        state_collection.update_one(
            {"_id": "global_state"},
            {"$set": {
                "wallet": STATE["wallet"], 
                "running": STATE["running"],
                "config": BOT_CONFIG  # <--- SAVING CONFIGURATION
            }},
            upsert=True
        )

def fetch_candles(tf_key, limit=300):
    tf = TF_MAP.get(tf_key, "1m")
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
        df["time"] = (df["time"] / 1000).astype(int)
        df["rsi"] = ta.rsi(df["close"], length=14)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching candles {tf}: {e}")
        return pd.DataFrame()

def last_price():
    try:
        return float(exchange.fetch_ticker(SYMBOL)["last"])
    except:
        return 0.0

# --- BACKGROUND BOT LOOP ---
def bot_loop():
    while True:
        try:
            current_price = last_price()
            
            # 1. Update PnL for Open Trades
            total_unrealized = 0.0
            trades_to_close = []
            
            for trade in STATE["openTrades"]:
                diff = current_price - trade["entryPrice"]
                if trade["side"] == "SHORT":
                    diff = -diff
                
                # Fee calculation
                fee_cost = (trade["size"] * current_price) * (trade.get("fee_rate", 0.0) / 100)
                trade["pnl"] = (diff * trade["size"]) - fee_cost
                
                total_unrealized += trade["pnl"]

                # Check SL/TP
                sl = trade.get("sl")
                tp = trade.get("tp")
                
                if trade["side"] == "LONG":
                    if sl and current_price <= sl: trades_to_close.append(trade)
                    if tp and current_price >= tp: trades_to_close.append(trade)
                elif trade["side"] == "SHORT":
                    if sl and current_price >= sl: trades_to_close.append(trade)
                    if tp and current_price <= tp: trades_to_close.append(trade)

            STATE["unrealized"] = total_unrealized
            
            # Process Auto Closes
            for t in trades_to_close:
                close_trade_internal(t, current_price, reason="TP/SL Hit")

            # 2. Automated Strategy Logic
            if STATE["running"]:
                s1_key, s2_key = BOT_CONFIG["stf1"], BOT_CONFIG["stf2"]
                l1_key, l2_key = BOT_CONFIG["ltf1"], BOT_CONFIG["ltf2"]
                
                df_stf_f = fetch_candles(s1_key)
                df_stf_s = fetch_candles(s2_key)
                df_ltf_f = fetch_candles(l1_key)
                df_ltf_s = fetch_candles(l2_key)
                
                if (len(df_stf_f) >= 2 and len(df_stf_s) >= 2 and 
                    not df_ltf_f.empty and not df_ltf_s.empty):
                    
                    stf_f_curr = df_stf_f["rsi"].iloc[-1]
                    stf_s_curr = df_stf_s["rsi"].iloc[-1]
                    stf_f_prev = df_stf_f["rsi"].iloc[-2]
                    stf_s_prev = df_stf_s["rsi"].iloc[-2]
                    
                    ltf_f_curr = df_ltf_f["rsi"].iloc[-1]
                    ltf_s_curr = df_ltf_s["rsi"].iloc[-1]
                    
                    is_bullish_trend = ltf_f_curr > ltf_s_curr
                    is_bearish_trend = ltf_f_curr < ltf_s_curr
                    
                    has_auto_trade = any(t.get('auto', False) for t in STATE["openTrades"])

                    if not has_auto_trade:
                        logic_str = ""
                        # LONG
                        if is_bullish_trend and (stf_f_prev <= stf_s_prev and stf_f_curr > stf_s_curr):
                            logic_str = f"{s1_key}>{s2_key} & {l1_key}>{l2_key}"
                            print(f"OPEN LONG: {logic_str}")
                            open_trade("LONG", current_price, logic_str)

                        # SHORT
                        elif is_bearish_trend and (stf_f_prev >= stf_s_prev and stf_f_curr < stf_s_curr):
                            logic_str = f"{s1_key}<{s2_key} & {l1_key}<{l2_key}"
                            print(f"OPEN SHORT: {logic_str}")
                            open_trade("SHORT", current_price, logic_str)

        except Exception as e:
            print(f"Bot Loop Error: {e}")
        
        time.sleep(2)

def open_trade(side, price, logic_desc="Manual"):
    sl = BOT_CONFIG["sl"]
    tp = BOT_CONFIG["tp"]
    
    new_trade = {
        "id": str(uuid.uuid4())[:8],
        "side": side,
        "size": BOT_CONFIG["qty"],
        "entryPrice": price,
        # SAVING ALL NEW FIELDS TO DB
        "sl": sl, 
        "tp": tp, 
        "fee_rate": BOT_CONFIG["fee"],
        "pnl": 0.0,
        "auto": True,
        "logic": logic_desc,
        "time": datetime.now().isoformat()
    }
    
    STATE["openTrades"].append(new_trade)
    
    if trades_collection:
        trades_collection.insert_one(new_trade.copy())

def close_trade_internal(trade, current_price, reason="Manual"):
    if trade in STATE["openTrades"]:
        STATE["openTrades"].remove(trade)
        
        # SAVING ALL COLUMNS TO HISTORY DB
        history_item = {
            "time": datetime.now().isoformat(),
            "side": trade["side"],
            "entryPrice": trade["entryPrice"],
            "exitPrice": current_price,      # Explicit Exit Price
            "price": current_price,          # Keep 'price' for frontend compatibility
            "qty": trade["size"],
            "realizedPnl": trade["pnl"],
            "fee_rate": trade.get("fee_rate", 0.0), # Save Fee
            "sl": trade.get("sl"),           # Save SL
            "tp": trade.get("tp"),           # Save TP
            "logic": trade.get("logic", "Manual"),
            "reason": reason
        }
        
        STATE["history"].append(history_item)
        STATE["wallet"] += trade["pnl"]
        
        if trades_collection and history_collection:
            trades_collection.delete_one({"id": trade["id"]})
            history_collection.insert_one(history_item.copy())
            update_db_state() # Save new wallet

threading.Thread(target=bot_loop, daemon=True).start()

# --- API ENDPOINTS ---

@app.get("/api/market")
def market():
    current_time = time.time()
    s1, s2 = BOT_CONFIG["stf1"], BOT_CONFIG["stf2"]
    l1, l2 = BOT_CONFIG["ltf1"], BOT_CONFIG["ltf2"]
    
    if CACHE["data"] is not None and (current_time - CACHE["last_update"] < 5):
        return CACHE["data"]

    def pack(df):
        if df.empty: return []
        return df[["time","open","high","low","close","rsi"]].to_dict("records")
        
    try:
        price = last_price()
        CACHE["data"] = {
            "price": price,
            "stf1": pack(fetch_candles(s1)),
            "stf2": pack(fetch_candles(s2)),
            "ltf1": pack(fetch_candles(l1)),
            "ltf2": pack(fetch_candles(l2)),
            "config": BOT_CONFIG,
            "state": STATE,
            "openTrades": STATE["openTrades"],
            "history": STATE["history"]
        }
        CACHE["last_update"] = current_time
        return CACHE["data"]
    except Exception as e:
        print(f"Market Data Error: {e}")
        if CACHE["data"]: return CACHE["data"]
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start")
def start(req: BotStartReq):
    # Update Config from Frontend
    BOT_CONFIG["stf1"] = req.stf1
    BOT_CONFIG["stf2"] = req.stf2
    BOT_CONFIG["ltf1"] = req.ltf1
    BOT_CONFIG["ltf2"] = req.ltf2
    BOT_CONFIG["qty"]  = req.qty
    BOT_CONFIG["fee"]  = req.fee
    BOT_CONFIG["sl"]   = req.sl
    BOT_CONFIG["tp"]   = req.tp
    
    STATE["running"] = True
    update_db_state() # Save 'running' AND 'config'
    return {"status": "started", "config": BOT_CONFIG}

@app.post("/api/stop")
def stop():
    STATE["running"] = False
    update_db_state() # Save 'stopped'
    return {"status": "stopped"}

@app.post("/api/manual/order")
def manual_order(order: ManualOrder):
    price = last_price()
    trade = {
        "id": str(uuid.uuid4())[:8],
        "side": order.side,
        "size": order.qty,
        "entryPrice": price,
        "sl": order.sl,
        "tp": order.tp,
        "fee_rate": 0.1, # Default manual fee
        "pnl": 0.0,
        "logic": "Manual",
        "time": datetime.now().isoformat()
    }
    
    STATE["openTrades"].append(trade)
    if trades_collection:
        trades_collection.insert_one(trade.copy())
        
    return {"status": "success", "trade": trade}

@app.post("/api/manual/close")
def manual_close(req: CloseTradeReq):
    trade_to_close = next((t for t in STATE["openTrades"] if t["id"] == req.id), None)
    if trade_to_close:
        close_trade_internal(trade_to_close, last_price(), reason="Manual Close")
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Trade not found")

@app.post("/api/manual/close-all")
def close_all():
    current_p = last_price()
    count = len(STATE["openTrades"])
    for t in list(STATE["openTrades"]):
        close_trade_internal(t, current_p, reason="Close All")
    return {"status": "success", "closed_count": count}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))

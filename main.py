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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- CONFIGURATION & DB SETUP ---
MONGO_URL = os.environ.get("MONGO_URL")

if not MONGO_URL:
    # Use a default URL only if running locally and MONGO_URL is not set
    print("❌ WARNING: MONGO_URL environment variable not found. Check Railway configuration.")
    # For local testing, ensure MongoDB is running on default port:
    # MONGO_URL = "mongodb://localhost:27017/" 
    # raise Exception("MONGO_URL is not set. Cannot connect to MongoDB.")

try:
    client = MongoClient(MONGO_URL)
    db = client.trading_bot # Database name
    
    # Collections for persistence
    state_collection = db.state_collection
    trades_collection = db.open_trades
    history_collection = db.history
    
    print("✅ MongoDB Client Initialized and Connected")

except Exception as e:
    print(f"❌ FATAL: Could not connect to MongoDB. Trades will NOT be persistent: {e}")
    # In a production environment, you might want to raise an exception here
    # For now, we continue but persistence will fail.

# --- CONFIGURATION ---
exchange = ccxt.binance({"enableRateLimit": True})
SYMBOL = "BTC/USDT"

# --- STATE MANAGEMENT ---
# We now have a global config to store the user's settings from the frontend
BOT_CONFIG = {
    "stf1": "1m", "stf2": "5m",
    "ltf1": "1h", "ltf2": "4h",
    "tp": None,   "sl": None,
    "fee": 0.1,   "qty": 0.01
}

STATE = {
    "running": False,
    "wallet": 1000.0,
    "unrealized": 0.0, 
    "openTrades": [],  
    "history": []      
}

# 30s is often not supported via REST OHLCV, so we map standard TFs
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
def fetch_candles(tf_key, limit=300):
    # Use the key to find the mapped value, default to 1m
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
                
                # Apply Fee Logic (Simulated impact on PnL)
                # Fee is usually paid on entry and exit. Here we subtract (size * price * fee%)
                fee_cost = (trade["size"] * current_price) * (trade.get("fee_rate", 0.0) / 100)
                trade["pnl"] = (diff * trade["size"]) - fee_cost
                
                total_unrealized += trade["pnl"]

                # Check SL/TP (Auto Close)
                sl = trade.get("sl")
                tp = trade.get("tp")
                
                # Long SL/TP
                if trade["side"] == "LONG":
                    if sl and current_price <= sl: trades_to_close.append(trade)
                    if tp and current_price >= tp: trades_to_close.append(trade)
                # Short SL/TP
                elif trade["side"] == "SHORT":
                    if sl and current_price >= sl: trades_to_close.append(trade)
                    if tp and current_price <= tp: trades_to_close.append(trade)

            STATE["unrealized"] = total_unrealized
            
            # Process Auto Closes
            for t in trades_to_close:
                close_trade_internal(t, current_price, reason="TP/SL Hit")

            # 2. Automated Strategy Logic
            if STATE["running"]:
                # Use Dynamic Config from Inputs
                s1_key, s2_key = BOT_CONFIG["stf1"], BOT_CONFIG["stf2"]
                l1_key, l2_key = BOT_CONFIG["ltf1"], BOT_CONFIG["ltf2"]
                
                df_stf_f = fetch_candles(s1_key)
                df_stf_s = fetch_candles(s2_key)
                df_ltf_f = fetch_candles(l1_key)
                df_ltf_s = fetch_candles(l2_key)
                
                if (len(df_stf_f) >= 2 and len(df_stf_s) >= 2 and 
                    not df_ltf_f.empty and not df_ltf_s.empty):
                    
                    # STF
                    stf_f_curr = df_stf_f["rsi"].iloc[-1]
                    stf_s_curr = df_stf_s["rsi"].iloc[-1]
                    stf_f_prev = df_stf_f["rsi"].iloc[-2]
                    stf_s_prev = df_stf_s["rsi"].iloc[-2]
                    
                    # LTF
                    ltf_f_curr = df_ltf_f["rsi"].iloc[-1]
                    ltf_s_curr = df_ltf_s["rsi"].iloc[-1]
                    
                    is_bullish_trend = ltf_f_curr > ltf_s_curr
                    is_bearish_trend = ltf_f_curr < ltf_s_curr
                    
                    has_auto_trade = any(t.get('auto', False) for t in STATE["openTrades"])

                    if not has_auto_trade:
                        # Logic String for Display
                        logic_str = ""
                        
                        # SCENARIO 1: LONG
                        if is_bullish_trend and (stf_f_prev <= stf_s_prev and stf_f_curr > stf_s_curr):
                            logic_str = f"{s1_key}>{s2_key} & {l1_key}>{l2_key}"
                            print(f"OPEN LONG: {logic_str}")
                            open_trade("LONG", current_price, logic_str)

                        # SCENARIO 2: SHORT
                        elif is_bearish_trend and (stf_f_prev >= stf_s_prev and stf_f_curr < stf_s_curr):
                            logic_str = f"{s1_key}<{s2_key} & {l1_key}<{l2_key}"
                            print(f"OPEN SHORT: {logic_str}")
                            open_trade("SHORT", current_price, logic_str)

        except Exception as e:
            print(f"Bot Loop Error: {e}")
        
        time.sleep(2)

def open_trade(side, price, logic_desc="Manual"):
    # Calculate SL/TP based on config if not manual
    sl = BOT_CONFIG["sl"]
    tp = BOT_CONFIG["tp"]
    
    # If SL/TP are percentages (simple logic), you might want to convert them to price
    # For now, we assume user inputs Absolute Price in the UI as per screenshot "Stop Loss Price"
    # Or if you want % based, you'd calculate: price * (1 - sl_pct/100) etc.
    # The prompt asked for "TP and SL option", usually implies Price or %. 
    # Since the manual input asks for "Price", we will assume the Config is also Price OR 
    # if it's small (<100) maybe it's percent? Let's assume Price for consistency with Manual.
    
    new_trade = {
        "id": str(uuid.uuid4())[:8],
        "side": side,
        "size": BOT_CONFIG["qty"],
        "entryPrice": price,
        "sl": sl, 
        "tp": tp, 
        "fee_rate": BOT_CONFIG["fee"],
        "pnl": 0.0,
        "auto": True,
        "logic": logic_desc, # <--- YELLOW CIRCLE REQ
        "time": datetime.now().isoformat()
    }
    STATE["openTrades"].append(new_trade)

def close_trade_internal(trade, current_price, reason="Manual"):
    if trade in STATE["openTrades"]:
        STATE["openTrades"].remove(trade)
        history_item = {
            "time": datetime.now().isoformat(),
            "side": trade["side"],
            "price": current_price,
            "qty": trade["size"],
            "realizedPnl": trade["pnl"],
            "logic": trade.get("logic", "Manual"),
            "reason": reason
        }
        STATE["history"].append(history_item)
        STATE["wallet"] += trade["pnl"]

threading.Thread(target=bot_loop, daemon=True).start()

# --- API ENDPOINTS ---

@app.get("/api/market")
def market():
    current_time = time.time()
    # Return data based on current CONFIG
    # We use the keys stored in BOT_CONFIG
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
            "config": BOT_CONFIG, # Send back config to confirm
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
    # Update Global Config
    BOT_CONFIG["stf1"] = req.stf1
    BOT_CONFIG["stf2"] = req.stf2
    BOT_CONFIG["ltf1"] = req.ltf1
    BOT_CONFIG["ltf2"] = req.ltf2
    BOT_CONFIG["qty"]  = req.qty
    BOT_CONFIG["fee"]  = req.fee
    BOT_CONFIG["sl"]   = req.sl
    BOT_CONFIG["tp"]   = req.tp
    
    STATE["running"] = True
    return {"status": "started", "config": BOT_CONFIG}

@app.post("/api/stop")
def stop():
    STATE["running"] = False
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
    # Iterate copy to avoid modification issues
    for t in list(STATE["openTrades"]):
        close_trade_internal(t, current_p, reason="Close All")
    return {"status": "success", "closed_count": count}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))

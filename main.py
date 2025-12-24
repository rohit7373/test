import os, time, threading, uuid
import pandas as pd
import pandas_ta as ta
import ccxt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional  # <--- ADDED THIS IMPORT
from datetime import datetime
import uvicorn

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

# --- STATE MANAGEMENT ---
STATE = {
    "running": False,
    "wallet": 1000.0,
    "unrealized": 0.0, 
    "openTrades": [],  
    "history": []      
}

TF_MAP = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m",
    "1h":"1h","4h":"4h","1d":"1d","1w":"1w"
}

# --- CACHING SYSTEM ---
CACHE = {
    "last_update": 0,
    "data": None
}

# --- DATA MODELS (FIXED) ---
class ManualOrder(BaseModel):
    side: str
    qty: float
    type: str
    # "Optional[float]" tells Python it is okay if these are null/empty
    sl: Optional[float] = None 
    tp: Optional[float] = None

class CloseTradeReq(BaseModel):
    id: str

# --- HELPER FUNCTIONS ---
def fetch_candles(tf, limit=300):
    tf = TF_MAP.get(tf, "1m")
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
        df["time"] = (df["time"] / 1000).astype(int)
        df["rsi"] = ta.rsi(df["close"], length=14)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching candles: {e}")
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
            for trade in STATE["openTrades"]:
                diff = current_price - trade["entryPrice"]
                if trade["side"] == "SHORT":
                    diff = -diff
                trade["pnl"] = diff * trade["size"]
                total_unrealized += trade["pnl"]
            
            STATE["unrealized"] = total_unrealized

            # 2. Automated Strategy Logic
            if STATE["running"]:
                df_f = fetch_candles("1m")
                df_s = fetch_candles("5m")
                
                if not df_f.empty and not df_s.empty:
                    rsi_f = df_f["rsi"].iloc[-1]
                    rsi_s = df_s["rsi"].iloc[-1]

                    # Only open auto-trade if none exist
                    has_auto_trade = any(t.get('auto', False) for t in STATE["openTrades"])

                    if not has_auto_trade:
                        if rsi_f > rsi_s: # LONG SIGNAL
                            new_trade = {
                                "id": str(uuid.uuid4())[:8],
                                "side": "LONG",
                                "size": 0.01,
                                "entryPrice": current_price,
                                "sl": None,
                                "tp": None,
                                "pnl": 0.0,
                                "auto": True,
                                "time": datetime.now().isoformat()
                            }
                            STATE["openTrades"].append(new_trade)
                            print("Auto LONG opened")

        except Exception as e:
            print(f"Bot Loop Error: {e}")
        
        time.sleep(2)

threading.Thread(target=bot_loop, daemon=True).start()

# --- API ENDPOINTS ---

@app.get("/api/market")
def market(stf1:str="1m", stf2:str="5m", ltf1:str="1h", ltf2:str="4h"):
    current_time = time.time()
    
    # Cache Check
    if CACHE["data"] is not None and (current_time - CACHE["last_update"] < 5):
        return CACHE["data"]

    def pack(df):
        if df.empty: return []
        return df[["time","open","high","low","close","rsi"]].to_dict("records")
        
    try:
        price = last_price()
        CACHE["data"] = {
            "price": price,
            "stf1": pack(fetch_candles(stf1)),
            "stf2": pack(fetch_candles(stf2)),
            "ltf1": pack(fetch_candles(ltf1)),
            "ltf2": pack(fetch_candles(ltf2)),
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
def start():
    STATE["running"] = True
    return {"status": "started"}

@app.post("/api/stop")
def stop():
    STATE["running"] = False
    return {"status": "stopped"}

# --- MANUAL TRADE ENDPOINTS ---

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
        "pnl": 0.0,
        "time": datetime.now().isoformat()
    }
    
    STATE["openTrades"].append(trade)
    print(f"Manual Trade Opened: {trade}")
    return {"status": "success", "trade": trade}

@app.post("/api/manual/close")
def manual_close(req: CloseTradeReq):
    trade_to_close = None
    for t in STATE["openTrades"]:
        if t["id"] == req.id:
            trade_to_close = t
            break
            
    if trade_to_close:
        STATE["openTrades"].remove(trade_to_close)
        
        history_item = {
            "time": datetime.now().isoformat(),
            "side": trade_to_close["side"],
            "price": last_price(),
            "qty": trade_to_close["size"],
            "realizedPnl": trade_to_close["pnl"]
        }
        STATE["history"].append(history_item)
        STATE["wallet"] += trade_to_close["pnl"]
        
        return {"status": "success", "message": "Trade closed"}
    
    raise HTTPException(status_code=404, detail="Trade not found")

@app.post("/api/manual/close-all")
def close_all():
    current_p = last_price()
    count = 0
    
    for t in STATE["openTrades"]:
        history_item = {
            "time": datetime.now().isoformat(),
            "side": t["side"],
            "price": current_p,
            "qty": t["size"],
            "realizedPnl": t["pnl"]
        }
        STATE["history"].append(history_item)
        STATE["wallet"] += t["pnl"]
        count += 1
        
    STATE["openTrades"] = []
    return {"status": "success", "closed_count": count}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))

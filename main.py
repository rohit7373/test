import os, time, threading, uuid, json
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

# --- CONFIGURATION ---
exchange = ccxt.binance({"enableRateLimit": True})
SYMBOL = "BTC/USDT"
DATA_FILE = "bot_data.json"

# --- STRATEGY SETTINGS (NEW) ---
# 0.01 = 1%, 0.02 = 2%
STOP_LOSS_PCT = 0.01   
TAKE_PROFIT_PCT = 0.02 

# --- STATE MANAGEMENT ---
STATE = {
    "running": False,
    "wallet": 1000.0,
    "unrealized": 0.0,
    "openTrades": [],
    "history": []
}

# --- PERSISTENCE FUNCTIONS ---
def save_state():
    try:
        data_to_save = {
            "running": STATE["running"],
            "wallet": STATE["wallet"],
            "openTrades": STATE["openTrades"],
            "history": STATE["history"]
        }
        with open(DATA_FILE, "w") as f:
            json.dump(data_to_save, f, indent=4)
    except Exception as e:
        print(f"Error saving data: {e}")

def load_state():
    global STATE
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                loaded = json.load(f)
                STATE.update(loaded)
                print("✅ Previous Trade Data Loaded!")
    except Exception as e:
        print(f"Error loading data: {e}")

load_state()

TF_MAP = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m",
    "1h":"1h","4h":"4h","1d":"1d","1w":"1w"
}

CACHE = { "last_update": 0, "data": None }

# --- DATA MODELS ---
class ManualOrder(BaseModel):
    side: str
    qty: float
    type: str
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

def close_trade_internal(trade, current_price, reason):
    """Closes a specific trade and saves state."""
    STATE["openTrades"].remove(trade)
    
    # Calculate Final PnL
    diff = current_price - trade["entryPrice"]
    if trade["side"] == "SHORT":
        diff = -diff
    final_pnl = diff * trade["size"]

    history_item = {
        "time": datetime.now().isoformat(),
        "side": trade["side"],
        "price": current_price,
        "qty": trade["size"],
        "realizedPnl": final_pnl,
        "reason": reason # e.g., "TP Hit", "SL Hit", "Manual"
    }
    STATE["history"].append(history_item)
    STATE["wallet"] += final_pnl
    save_state()
    print(f"🚫 Trade Closed ({reason}): {final_pnl:.2f}")

# --- BACKGROUND BOT LOOP ---
def bot_loop():
    while True:
        try:
            current_price = last_price()
            
            # --- 1. MONITOR OPEN TRADES (SL/TP CHECK) ---
            total_unrealized = 0.0
            trades_to_close = []

            for trade in STATE["openTrades"]:
                # Update PnL
                diff = current_price - trade["entryPrice"]
                if trade["side"] == "SHORT":
                    diff = -diff
                trade["pnl"] = diff * trade["size"]
                total_unrealized += trade["pnl"]

                # Check SL / TP
                if trade["sl"] and trade["tp"]:
                    if trade["side"] == "LONG":
                        if current_price <= trade["sl"]:
                            trades_to_close.append((trade, "SL Hit"))
                        elif current_price >= trade["tp"]:
                            trades_to_close.append((trade, "TP Hit"))
                    
                    elif trade["side"] == "SHORT":
                        if current_price >= trade["sl"]:
                            trades_to_close.append((trade, "SL Hit"))
                        elif current_price <= trade["tp"]:
                            trades_to_close.append((trade, "TP Hit"))
            
            # Close flagged trades
            for t, reason in trades_to_close:
                close_trade_internal(t, current_price, reason)
            
            STATE["unrealized"] = total_unrealized

            # --- 2. AUTOMATED STRATEGY LOGIC ---
            if STATE["running"]:
                df_stf_f = fetch_candles("1m")
                df_stf_s = fetch_candles("5m")
                df_ltf_f = fetch_candles("1h")
                df_ltf_s = fetch_candles("4h")
                
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
                        # LONG ENTRY
                        if is_bullish_trend and (stf_f_prev <= stf_s_prev and stf_f_curr > stf_s_curr):
                            print("🚀 AUTO LONG SIGNAL")
                            # Calculate SL/TP for LONG
                            sl_price = current_price * (1 - STOP_LOSS_PCT)
                            tp_price = current_price * (1 + TAKE_PROFIT_PCT)
                            open_trade_internal("LONG", current_price, 0.01, sl_price, tp_price, True)

                        # SHORT ENTRY
                        elif is_bearish_trend and (stf_f_prev >= stf_s_prev and stf_f_curr < stf_s_curr):
                            print("🚀 AUTO SHORT SIGNAL")
                            # Calculate SL/TP for SHORT
                            sl_price = current_price * (1 + STOP_LOSS_PCT)
                            tp_price = current_price * (1 - TAKE_PROFIT_PCT)
                            open_trade_internal("SHORT", current_price, 0.01, sl_price, tp_price, True)

        except Exception as e:
            print(f"Bot Loop Error: {e}")
        
        time.sleep(2)

def open_trade_internal(side, price, qty, sl, tp, is_auto=False):
    new_trade = {
        "id": str(uuid.uuid4())[:8],
        "side": side,
        "size": qty,
        "entryPrice": price,
        "sl": sl, 
        "tp": tp, 
        "pnl": 0.0,
        "auto": is_auto,
        "time": datetime.now().isoformat()
    }
    STATE["openTrades"].append(new_trade)
    save_state()

threading.Thread(target=bot_loop, daemon=True).start()

# --- API ENDPOINTS ---

@app.get("/api/market")
def market(stf1:str="1m", stf2:str="5m", ltf1:str="1h", ltf2:str="4h"):
    current_time = time.time()
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
    save_state()
    return {"status": "started"}

@app.post("/api/stop")
def stop():
    STATE["running"] = False
    save_state()
    return {"status": "stopped"}

@app.post("/api/manual/order")
def manual_order(order: ManualOrder):
    price = last_price()
    
    trade = {
        "id": str(uuid.uuid4())[:8],
        "side": order.side,
        "size": order.qty,
        "entryPrice": price,
        "sl": order.sl, # Uses the manual input
        "tp": order.tp, # Uses the manual input
        "pnl": 0.0,
        "auto": False,
        "time": datetime.now().isoformat()
    }
    
    STATE["openTrades"].append(trade)
    print(f"Manual Trade Opened: {trade}")
    save_state()
    return {"status": "success", "trade": trade}

@app.post("/api/manual/close")
def manual_close(req: CloseTradeReq):
    price = last_price()
    trade_to_close = None
    for t in STATE["openTrades"]:
        if t["id"] == req.id:
            trade_to_close = t
            break
            
    if trade_to_close:
        close_trade_internal(trade_to_close, price, "Manual Close")
        return {"status": "success", "message": "Trade closed"}
    
    raise HTTPException(status_code=404, detail="Trade not found")

@app.post("/api/manual/close-all")
def close_all():
    current_p = last_price()
    count = len(STATE["openTrades"])
    
    # Create a copy of the list to iterate safely while modifying
    for t in list(STATE["openTrades"]):
        close_trade_internal(t, current_p, "Manual Close All")
        
    return {"status": "success", "closed_count": count}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))

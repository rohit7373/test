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

# --- FEE & STRATEGY SETTINGS ---
FEE_RATE = 0.00025   # 0.025% Fee
TARGET_NET_PROFIT = 0.004 # 0.4% Actual Profit
MAX_NET_LOSS = 0.003      # 0.3% Actual Loss

# --- STATE MANAGEMENT ---
STATE = {
    "running": False,
    "wallet": 1000.0,
    "unrealized": 0.0,
    "openTrades": [],
    "history": []
}

# --- PERSISTENCE ---
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
                print("✅ Data Loaded Successfully")
    except Exception as e:
        print(f"Error loading data: {e}")

load_state()

TF_MAP = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m",
    "1h":"1h","4h":"4h","1d":"1d","1w":"1w"
}

CACHE = { "last_update": 0, "data": None }

# --- MODELS ---
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

# --- PNL & PRICE CALCULATION LOGIC ---

def calculate_net_pnl(side, entry_price, exit_price, qty):
    """
    Calculates ACTUAL PnL after deducting 0.025% fee from both Entry and Exit.
    """
    turnover_entry = entry_price * qty
    turnover_exit = exit_price * qty
    
    fee_entry = turnover_entry * FEE_RATE
    fee_exit = turnover_exit * FEE_RATE
    
    if side == "LONG":
        gross_pnl = turnover_exit - turnover_entry
    else: # SHORT
        gross_pnl = turnover_entry - turnover_exit
        
    net_pnl = gross_pnl - fee_entry - fee_exit
    return net_pnl

def calculate_tp_sl_prices(side, entry_price):
    """
    Calculates TP/SL prices to ensure ACTUAL Net Profit of 0.4% 
    and ACTUAL Net Loss of 0.3% after fees.
    """
    # Formulas derived to account for fees on both ends:
    # Net = (Exit - Entry) - (Entry*Fee) - (Exit*Fee)  [simplified]
    
    if side == "LONG":
        # Target TP: Entry * (1 + Fee + Target) / (1 - Fee)
        tp_price = entry_price * (1 + FEE_RATE + TARGET_NET_PROFIT) / (1 - FEE_RATE)
        # Target SL: Entry * (1 + Fee - MaxLoss) / (1 - Fee)
        sl_price = entry_price * (1 + FEE_RATE - MAX_NET_LOSS) / (1 - FEE_RATE)
        
    else: # SHORT
        # Target TP: Entry * (1 - Fee - Target) / (1 + Fee)
        tp_price = entry_price * (1 - FEE_RATE - TARGET_NET_PROFIT) / (1 + FEE_RATE)
        # Target SL: Entry * (1 - Fee + MaxLoss) / (1 + Fee)
        sl_price = entry_price * (1 - FEE_RATE + MAX_NET_LOSS) / (1 + FEE_RATE)
        
    return tp_price, sl_price

# --- BOT LOOP ---
def bot_loop():
    while True:
        try:
            current_p = last_price()
            
            # 1. Update Open Trades PnL & Check TP/SL
            total_unrealized = 0.0
            trades_to_close = []
            
            for trade in STATE["openTrades"]:
                # Calculate Net PnL Live
                net_pnl = calculate_net_pnl(trade["side"], trade["entryPrice"], current_p, trade["size"])
                trade["pnl"] = net_pnl
                total_unrealized += net_pnl
                
                # Check Auto-Close Conditions (TP/SL)
                # Note: We use the pre-calculated TP/SL prices which already account for fees
                if trade["tp"] and trade["sl"]:
                    if trade["side"] == "LONG":
                        if current_p >= trade["tp"]: trades_to_close.append((trade, "TP"))
                        elif current_p <= trade["sl"]: trades_to_close.append((trade, "SL"))
                    else: # SHORT
                        if current_p <= trade["tp"]: trades_to_close.append((trade, "TP"))
                        elif current_p >= trade["sl"]: trades_to_close.append((trade, "SL"))

            STATE["unrealized"] = total_unrealized

            # Close Hit Trades
            for t, reason in trades_to_close:
                close_trade_internal(t["id"], current_p, reason)

            # 2. Strategy Logic
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
                    
                    is_bullish = ltf_f_curr > ltf_s_curr
                    is_bearish = ltf_f_curr < ltf_s_curr
                    
                    has_auto = any(t.get('auto', False) for t in STATE["openTrades"])

                    if not has_auto:
                        if is_bullish and (stf_f_prev <= stf_s_prev and stf_f_curr > stf_s_curr):
                            print("AUTO LONG TRIGGER")
                            open_trade_internal("LONG", current_p, 0.01, True)

                        elif is_bearish and (stf_f_prev >= stf_s_prev and stf_f_curr < stf_s_curr):
                            print("AUTO SHORT TRIGGER")
                            open_trade_internal("SHORT", current_p, 0.01, True)

        except Exception as e:
            print(f"Bot Loop Error: {e}")
        
        time.sleep(2)

def open_trade_internal(side, price, qty, is_auto=False):
    # Auto-Calculate TP/SL based on User's Net Profit Rules
    tp_price, sl_price = calculate_tp_sl_prices(side, price)
    
    new_trade = {
        "id": str(uuid.uuid4())[:8],
        "side": side,
        "size": qty,
        "entryPrice": price,
        "sl": sl_price, 
        "tp": tp_price, 
        "pnl": 0.0 - (price * qty * FEE_RATE * 2), # Initial PnL is negative (entry fee + est exit fee)
        "auto": is_auto,
        "time": datetime.now().isoformat()
    }
    STATE["openTrades"].append(new_trade)
    save_state()

def close_trade_internal(trade_id, exit_price, reason="MANUAL"):
    trade_to_close = None
    for t in STATE["openTrades"]:
        if t["id"] == trade_id:
            trade_to_close = t
            break
            
    if trade_to_close:
        STATE["openTrades"].remove(trade_to_close)
        
        # Calculate Final Net PnL
        final_pnl = calculate_net_pnl(trade_to_close["side"], trade_to_close["entryPrice"], exit_price, trade_to_close["size"])
        
        history_item = {
            "time": datetime.now().isoformat(),
            "side": trade_to_close["side"],
            "price": exit_price,
            "qty": trade_to_close["size"],
            "realizedPnl": final_pnl,
            "reason": reason
        }
        STATE["history"].append(history_item)
        STATE["wallet"] += final_pnl
        save_state()
        print(f"Trade Closed ({reason}): PnL {final_pnl:.4f}")

threading.Thread(target=bot_loop, daemon=True).start()

# --- API ---
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
    
    # If user provided Manual SL/TP, use them. 
    # Otherwise, use the Auto-Calc based on 0.4% / 0.3% Logic
    if order.sl and order.tp:
        final_sl, final_tp = order.sl, order.tp
    else:
        auto_tp, auto_sl = calculate_tp_sl_prices(order.side, price)
        # Use user's value if provided, else use auto
        final_tp = order.tp if order.tp else auto_tp
        final_sl = order.sl if order.sl else auto_sl

    # Initial PnL = Negative (Entry Fee + Estimated Exit Fee)
    est_fee = (price * order.qty * FEE_RATE) * 2 
    
    trade = {
        "id": str(uuid.uuid4())[:8],
        "side": order.side,
        "size": order.qty,
        "entryPrice": price,
        "sl": final_sl,
        "tp": final_tp,
        "pnl": -est_fee, 
        "auto": False,
        "time": datetime.now().isoformat()
    }
    
    STATE["openTrades"].append(trade)
    save_state()
    return {"status": "success", "trade": trade}

@app.post("/api/manual/close")
def manual_close(req: CloseTradeReq):
    close_trade_internal(req.id, last_price(), "MANUAL")
    return {"status": "success"}

@app.post("/api/manual/close-all")
def close_all():
    current_p = last_price()
    count = 0
    # Create a copy to iterate safely while modifying
    for t in list(STATE["openTrades"]):
        close_trade_internal(t["id"], current_p, "CLOSE_ALL")
        count += 1
    return {"status": "success", "closed_count": count}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))

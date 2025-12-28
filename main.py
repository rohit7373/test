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
# Full list of valid TFs
VALID_TFS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"]


# --- DUAL BOT CONFIGURATION ---
# Default logic settings for a LONG (BUY) trade
BUY_CONFIG = {
    "stf1": "5m", "stf2": "15m",
    "ltf1": "1h", "ltf2": "4h",
    "stf_logic": ">",       # STF1 > STF2 (Buy Cross)
    "ltf_logic": ">",       # LTF1 > LTF2 (Buy Filter)
    "entry_mode": "BOTH",   # Options: "STF_ONLY", "LTF_ONLY", "BOTH"
}
# Default logic settings for a SHORT (SELL) trade
SELL_CONFIG = {
    "stf1": "5m", "stf2": "15m",
    "ltf1": "1h", "ltf2": "4h",
    "stf_logic": "<",       # STF1 < STF2 (Sell Cross)
    "ltf_logic": "<",       # LTF1 < LTF2 (Sell Filter)
    "entry_mode": "BOTH",
}
# General Settings (applies to both Buy and Sell)
GENERAL_CONFIG = {
    "qty": 0.01,
    "fee": 0.1,             # Fee rate in percent
    "tp": None,             # Auto TP (price)
    "sl": None,             # Auto SL (price)
    # Default chart display TFs (used if not loaded from DB)
    "stf1": "5m", "stf2": "15m", "ltf1": "1h", "ltf2": "4h",
}


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

# Caching for market data to reduce CCXT API calls
CACHE = {"last_update": 0, "data": None}


# --- Pydantic Models for Requests ---
class BotStartReq(BaseModel):
    # Buy Config
    buy_stf1: str; buy_stf2: str; buy_ltf1: str; buy_ltf2: str
    buy_stf_logic: str; buy_ltf_logic: str; buy_entry_mode: str
    # Sell Config
    sell_stf1: str; sell_stf2: str; sell_ltf1: str; sell_ltf2: str
    sell_stf_logic: str; sell_ltf_logic: str; sell_entry_mode: str
    # General Config
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
    """Saves state and configurations to MongoDB."""
    global BUY_CONFIG, SELL_CONFIG, GENERAL_CONFIG
    
    # NOTE: GENERAL_CONFIG fields for chart display TFs are updated in /api/market
    
    if state_collection is not None:
        try:
            state_collection.update_one(
                {"_id": "global_state"},
                {"$set": {
                    "wallet": STATE["wallet"], 
                    "running": STATE["running"], 
                    "general_config": GENERAL_CONFIG,
                    "buy_config": BUY_CONFIG,
                    "sell_config": SELL_CONFIG
                }},
                upsert=True
            )
        except Exception as e:
            print(f"DB Update Failed: {e}")

def fetch_candles(tf_key, limit=300):
    """Fetch and prepare OHLCV data with RSI, returning a DataFrame."""
    if tf_key not in VALID_TFS: return pd.DataFrame()
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, tf_key, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = df['time'] // 1000  # Convert milliseconds to seconds
        df.ta.rsi(length=14, append=True)
        df.rename(columns={'RSI_14': 'rsi'}, inplace=True)
        df = df[['time', 'open', 'high', 'low', 'close', 'rsi']].dropna()
        return df
    except Exception as e:
        # print(f"CCXT Fetch Error for {tf_key}: {e}") # Suppress frequent errors
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
        fee_rate = t.get('fee_rate', GENERAL_CONFIG['fee']) / 100.0
        
        # Estimate total fee cost (entry + exit)
        total_fee_cost = (entry * size * fee_rate) + (current_price * size * fee_rate)
        
        if t['side'] == 'LONG':
            unrealized_pnl = (current_price - entry) * size - total_fee_cost
        else: # SHORT
            unrealized_pnl = (entry - current_price) * size - total_fee_cost

        # Update PnL in the open trade state (for API response)
        t['pnl'] = unrealized_pnl
        pnl += unrealized_pnl
        
    STATE["unrealized"] = pnl
    return pnl

def check_condition(val1, op, val2):
    """Checks the current filter condition (RSI1 vs RSI2)."""
    if op == ">": return val1 > val2
    if op == "<": return val1 < val2
    return False

def check_cross(prev1, curr1, op, prev2, curr2):
    """Checks for a cross signal (RSI1 crosses RSI2)."""
    if op == ">":
        # Bullish/Long Cross: prev1 <= prev2 AND curr1 > curr2
        return (prev1 <= prev2) and (curr1 > curr2)
    if op == "<":
        # Bearish/Short Cross: prev1 >= prev2 AND curr1 < curr2
        return (prev1 >= prev2) and (curr1 < curr2)
    return False

def open_trade(side, price, logic="Manual"):
    """Opens a new trade and saves it to state and DB."""
    t = {
        "id": str(uuid.uuid4())[:8], "side": side, "size": float(GENERAL_CONFIG["qty"]),
        "entryPrice": float(price), 
        # Use current global SL/TP as trade-specific limits
        "sl": GENERAL_CONFIG.get("sl"), "tp": GENERAL_CONFIG.get("tp"),
        "fee_rate": float(GENERAL_CONFIG["fee"]), "pnl": 0.0, "auto": True, "logic": logic,
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
        fee_rate = t.get('fee_rate', GENERAL_CONFIG['fee']) / 100.0

        # Total fee calculation (entry fee + exit fee)
        total_fee_cost = (entry * size * fee_rate) + (price * size * fee_rate)
        
        if t['side'] == 'LONG':
            realized_pnl = (price - entry) * size - total_fee_cost
        else: # SHORT
            realized_pnl = (entry - price) * size - total_fee_cost

        STATE["wallet"] += realized_pnl
        
        # History record
        h = {
            "time": datetime.now().isoformat(), "side": t["side"],
            "entryPrice": t["entryPrice"], "exitPrice": float(price), "qty": t["size"],
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

# --- CORE ENTRY LOGIC CHECK ---
def check_entry_signal(config, df_stf_f, df_stf_s, df_ltf_f, df_ltf_s, side):
    """Checks for the entry signal based on a specific (Buy or Sell) config."""
    min_len = 2
    if any(len(df) < min_len for df in [df_stf_f, df_stf_s, df_ltf_f, df_ltf_s]):
        return False, None

    try:
        # Get current and previous RSI values
        stf_f, stf_s = df_stf_f["rsi"].iloc[-1], df_stf_s["rsi"].iloc[-1]
        stf_f_prev, stf_s_prev = df_stf_f["rsi"].iloc[-2], df_stf_s["rsi"].iloc[-2]
        ltf_f, ltf_s = df_ltf_f["rsi"].iloc[-1], df_ltf_s["rsi"].iloc[-1]
        ltf_f_prev, ltf_s_prev = df_ltf_f["rsi"].iloc[-2], df_ltf_s["rsi"].iloc[-2]
        
        if math.isnan(stf_f) or math.isnan(ltf_f): return False, None
        
        stf_op = config["stf_logic"]
        ltf_op = config["ltf_logic"]
        entry_mode = config["entry_mode"]
        
        # --- Evaluate Conditions ---
        
        # 1. STF Entry Cross Check
        stf_cross = check_cross(stf_f_prev, stf_f, stf_op, stf_s_prev, stf_s)

        # 2. LTF Filter Check (LTF is in the direction of the trade)
        ltf_filter = check_condition(ltf_f, ltf_op, ltf_s)
        
        # 3. LTF Cross Check (Used for LTF_ONLY mode)
        ltf_cross = check_cross(ltf_f_prev, ltf_f, ltf_op, ltf_s_prev, ltf_s)

        
        trade_allowed = False
        
        if entry_mode == "BOTH":
            trade_allowed = stf_cross and ltf_filter
        elif entry_mode == "STF_ONLY":
            trade_allowed = stf_cross
        elif entry_mode == "LTF_ONLY":
            trade_allowed = ltf_cross

        if trade_allowed:
            logic_str = f"{side} | Mode:{entry_mode} | STF:{config['stf1']}{stf_op}{config['stf2']} | LTF:{config['ltf1']}{ltf_op}{config['ltf2']}"
        else:
            logic_str = None
        
        return trade_allowed, logic_str

    except Exception as e:
        print(f"Signal Check Error for {side}: {e}")
        return False, None


def check_exit_signal(current_price):
    """Checks for TP/SL signals based on current price for all open trades."""
    trades_to_close = []
    
    for t in STATE["openTrades"]:
        exit_reason = None
        
        # Use General SL/TP if trade-specific ones are not set or 0/None
        sl = t.get('sl') if t.get('sl') is not None and t.get('sl') > 0 else GENERAL_CONFIG.get('sl')
        tp = t.get('tp') if t.get('tp') is not None and t.get('tp') > 0 else GENERAL_CONFIG.get('tp')
        
        if sl is not None and sl > 0:
            if t['side'] == 'LONG' and current_price <= sl: exit_reason = "SL Hit"
            elif t['side'] == 'SHORT' and current_price >= sl: exit_reason = "SL Hit"

        if exit_reason is None and tp is not None and tp > 0:
            if t['side'] == 'LONG' and current_price >= tp: exit_reason = "TP Hit"
            elif t['side'] == 'SHORT' and current_price <= tp: exit_reason = "TP Hit"
        
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
                # This prevents rapid-fire trades and conflicting auto trades
                if not any(t.get('auto') for t in STATE["openTrades"]):
                    
                    # Consolidate TFs to fetch only unique ones once
                    tfs_to_fetch = set([
                        BUY_CONFIG["stf1"], BUY_CONFIG["stf2"], BUY_CONFIG["ltf1"], BUY_CONFIG["ltf2"],
                        SELL_CONFIG["stf1"], SELL_CONFIG["stf2"], SELL_CONFIG["ltf1"], SELL_CONFIG["ltf2"],
                    ])
                    # Fetch and cache all required dataframes
                    dfs = {tf: fetch_candles(tf, limit=50) for tf in tfs_to_fetch}
                    
                    # --- Check BUY Signal ---
                    buy_trigger, buy_logic = check_entry_signal(
                        BUY_CONFIG, 
                        dfs.get(BUY_CONFIG["stf1"], pd.DataFrame()), dfs.get(BUY_CONFIG["stf2"], pd.DataFrame()),
                        dfs.get(BUY_CONFIG["ltf1"], pd.DataFrame()), dfs.get(BUY_CONFIG["ltf2"], pd.DataFrame()),
                        "LONG"
                    )
                    
                    # --- Check SELL Signal ---
                    sell_trigger, sell_logic = check_entry_signal(
                        SELL_CONFIG, 
                        dfs.get(SELL_CONFIG["stf1"], pd.DataFrame()), dfs.get(SELL_CONFIG["stf2"], pd.DataFrame()),
                        dfs.get(SELL_CONFIG["ltf1"], pd.DataFrame()), dfs.get(SELL_CONFIG["ltf2"], pd.DataFrame()),
                        "SHORT"
                    )

                    if buy_trigger and not sell_trigger:
                        open_trade("LONG", current_price, buy_logic)
                    elif sell_trigger and not buy_trigger:
                        open_trade("SHORT", current_price, sell_logic)
                    # If both are true, no trade is opened (conflicting signals)

            update_db() # Save State (wallet/running/config)
                
            time.sleep(5) # Poll every 5 seconds

        except Exception as e:
            print(f"Bot Loop Error: {e}")
            time.sleep(5)

# --- DB CONNECTION (Load Dual Config) ---

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
            
            # Load Dual Configs (New Format)
            if "buy_config" in saved: BUY_CONFIG.update(saved["buy_config"])
            if "sell_config" in saved: SELL_CONFIG.update(saved["sell_config"])
            if "general_config" in saved: GENERAL_CONFIG.update(saved["general_config"])
            
            # ------------------------------------------------------------------
            # REVISED MIGRATION LOGIC (if coming from a single config older version)
            # ------------------------------------------------------------------
            elif "config" in saved: 
                old_cfg = saved["config"]
                
                # 1. Migrate General Settings
                GENERAL_CONFIG_KEYS = ["qty", "fee", "tp", "sl"]
                for k in GENERAL_CONFIG_KEYS:
                    if k in old_cfg: GENERAL_CONFIG[k] = old_cfg[k]
                
                # 2. Migrate Logic Settings (Applies to both Buy and Sell)
                # Note: We assume the old logic applies to both buy and sell sides for simplicity
                LOGIC_CONFIG_KEYS = ["stf1", "stf2", "ltf1", "ltf2", "stf_logic", "ltf_logic", "entry_mode"]
                logic_update = {}
                for k in LOGIC_CONFIG_KEYS:
                    if k in old_cfg: logic_update[k] = old_cfg[k]
                    
                BUY_CONFIG.update(logic_update)
                SELL_CONFIG.update(logic_update)
            # ------------------------------------------------------------------
            # END OF REVISED MIGRATION LOGIC
            # ------------------------------------------------------------------

        else:
            # Initialize state if no global state exists
            state_collection.insert_one({"_id": "global_state", "wallet": 1000.0, "running": False, "buy_config": BUY_CONFIG, "sell_config": SELL_CONFIG, "general_config": GENERAL_CONFIG})

        # Load Trades
        STATE["openTrades"] = clean_data(list(trades_collection.find({})))
        STATE["history"] = clean_data(list(history_collection.find().sort("time", -1).limit(100)))

    except Exception as e:
        print(f"❌ DB Error: {e}")

threading.Thread(target=bot_loop, daemon=True).start()

# --- API ENDPOINTS ---

@app.get("/api/config")
async def get_initial_config():
    """Serves all configurations once for initial UI population."""
    return JSONResponse(content=clean_data({
        "buy_config": BUY_CONFIG,
        "sell_config": SELL_CONFIG,
        "general_config": GENERAL_CONFIG,
        "valid_tfs": VALID_TFS
    }))
    
@app.get("/api/market")
def market(
    stf1_display: Optional[str] = GENERAL_CONFIG["stf1"], 
    stf2_display: Optional[str] = GENERAL_CONFIG["stf2"],
    ltf1_display: Optional[str] = GENERAL_CONFIG["ltf1"], 
    ltf2_display: Optional[str] = GENERAL_CONFIG["ltf2"],
    ):
    """
    Sends volatile data (price, state, trades, history) and 
    dataframes for the TFs requested by the frontend for charting.
    """
    try:
        # 1. Update GENERAL_CONFIG with the TFs requested by the frontend for display
        # This ensures the DB saves the last viewed TFs
        GENERAL_CONFIG["stf1"] = stf1_display
        GENERAL_CONFIG["stf2"] = stf2_display
        GENERAL_CONFIG["ltf1"] = ltf1_display
        GENERAL_CONFIG["ltf2"] = ltf2_display
        
        current_tfs = set([stf1_display, stf2_display, ltf1_display, ltf2_display])
        
        # 2. Fetch data for the four current display TFs
        data_frames = {tf: fetch_candles(tf, limit=300) for tf in current_tfs if tf is not None}
        
        data = {
            "price": last_price(),
            "stf1": pack_df_to_list(data_frames.get(stf1_display, pd.DataFrame())),
            "stf2": pack_df_to_list(data_frames.get(stf2_display, pd.DataFrame())),
            "ltf1": pack_df_to_list(data_frames.get(ltf1_display, pd.DataFrame())),
            "ltf2": pack_df_to_list(data_frames.get(ltf2_display, pd.DataFrame())),
            "config": {
                "buy_config": BUY_CONFIG,
                "sell_config": SELL_CONFIG,
                "general_config": GENERAL_CONFIG,
            },
            "state": STATE,
            "openTrades": STATE["openTrades"],
            "history": STATE["history"]
        }
        
        clean_response = clean_data(data)
        return JSONResponse(content=clean_response)

    except Exception as e:
        print(f"Market Endpoint Error: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/start")
def start(req: BotStartReq):
    try:
        if STATE["running"]:
            raise HTTPException(status_code=400, detail="Bot is already running")
            
        # 1. Validate General Config
        if req.qty <= 0: raise HTTPException(status_code=400, detail="Quantity must be greater than 0")

        # 2. Update General Config
        global GENERAL_CONFIG
        GENERAL_CONFIG.update({
            "qty": req.qty, "fee": req.fee, 
            "tp": req.tp, "sl": req.sl
        })
        
        # 3. Update Buy Config
        global BUY_CONFIG
        if req.buy_stf_logic not in (">", "<") or req.buy_ltf_logic not in (">", "<"):
             raise HTTPException(status_code=400, detail="Buy logic operators must be '>' or '<'")
        if req.buy_entry_mode not in ("STF_ONLY", "LTF_ONLY", "BOTH"):
             raise HTTPException(status_code=400, detail="Buy entry mode must be 'STF_ONLY', 'LTF_ONLY', or 'BOTH'")
        BUY_CONFIG.update({
            "stf1": req.buy_stf1, "stf2": req.buy_stf2, "ltf1": req.buy_ltf1, "ltf2": req.buy_ltf2,
            "stf_logic": req.buy_stf_logic, "ltf_logic": req.buy_ltf_logic, "entry_mode": req.buy_entry_mode,
        })

        # 4. Update Sell Config
        global SELL_CONFIG
        if req.sell_stf_logic not in (">", "<") or req.sell_ltf_logic not in (">", "<"):
             raise HTTPException(status_code=400, detail="Sell logic operators must be '>' or '<'")
        if req.sell_entry_mode not in ("STF_ONLY", "LTF_ONLY", "BOTH"):
             raise HTTPException(status_code=400, detail="Sell entry mode must be 'STF_ONLY', 'LTF_ONLY', or 'BOTH'")
        SELL_CONFIG.update({
            "stf1": req.sell_stf1, "stf2": req.sell_stf2, "ltf1": req.sell_ltf1, "ltf2": req.sell_ltf2,
            "stf_logic": req.sell_stf_logic, "ltf_logic": req.sell_ltf_logic, "entry_mode": req.sell_entry_mode,
        })
        
        STATE["running"] = True
        update_db()
        return clean_data({"status": "started", "buy_config": BUY_CONFIG, "sell_config": SELL_CONFIG, "general_config": GENERAL_CONFIG})
    except Exception as e:
        return JSONResponse(status_code=500, detail=str(e))

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
        if not price: raise HTTPException(status_code=500, detail="Could not fetch current price.")
             
        t = {
            "id": str(uuid.uuid4())[:8], "side": order.side, "size": float(order.qty),
            "entryPrice": price, "sl": order.sl, "tp": order.tp, "fee_rate": GENERAL_CONFIG["fee"],
            "pnl": 0.0, "logic": "Manual", "auto": False, "time": datetime.now().isoformat()
        }
        STATE["openTrades"].append(t)
        
        if trades_collection is not None: 
            try: trades_collection.insert_one(t.copy())
            except: pass
            
        update_db()
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

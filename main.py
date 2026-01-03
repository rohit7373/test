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

# --- DUAL BOT CONFIGURATION (RSI) ---
BUY_CONFIG = {
    "stf1": "5m", "stf2": "15m", "ltf1": "1h", "ltf2": "4h",
    "stf_logic": ">", "ltf_logic": ">", "entry_mode": "BOTH",
}
SELL_CONFIG = {
    "stf1": "5m", "stf2": "15m", "ltf1": "1h", "ltf2": "4h",
    "stf_logic": "<", "ltf_logic": "<", "entry_mode": "BOTH",
}

# --- NEW: LIQUIDITY CONFIGURATION ---
LIQ_CONFIG = {
    "enabled": False,       # Master switch for Liquidity Strategy
    "ext_len": 20,          # External Structure Length
    "int_len": 5,           # Internal Structure Length
    "trade_int": False,     # Trade Internal Sweeps?
    "tf": "5m"              # Timeframe to scan for liquidity (default 5m)
}

# General Settings
GENERAL_CONFIG = {
    "qty": 0.01,
    "fee": 0.1,
    "tp": None, "sl": None,
    "stf1": "5m", "stf2": "15m", "ltf1": "1h", "ltf2": "4h", # Display defaults
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

# --- Pydantic Models ---
class BotStartReq(BaseModel):
    # Buy Config
    buy_stf1: str; buy_stf2: str; buy_ltf1: str; buy_ltf2: str
    buy_stf_logic: str; buy_ltf_logic: str; buy_entry_mode: str
    # Sell Config
    sell_stf1: str; sell_stf2: str; sell_ltf1: str; sell_ltf2: str
    sell_stf_logic: str; sell_ltf_logic: str; sell_entry_mode: str
    # Liquidity Config (NEW)
    liq_enabled: Optional[bool] = False
    liq_ext_len: Optional[int] = 20
    liq_int_len: Optional[int] = 5
    liq_trade_int: Optional[bool] = False
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
    global BUY_CONFIG, SELL_CONFIG, GENERAL_CONFIG, LIQ_CONFIG
    
    if state_collection is not None:
        try:
            state_collection.update_one(
                {"_id": "global_state"},
                {"$set": {
                    "wallet": STATE["wallet"], 
                    "running": STATE["running"], 
                    "general_config": GENERAL_CONFIG,
                    "buy_config": BUY_CONFIG,
                    "sell_config": SELL_CONFIG,
                    "liq_config": LIQ_CONFIG
                }},
                upsert=True
            )
        except Exception as e:
            print(f"DB Update Failed: {e}")

def fetch_candles(tf_key, limit=300):
    if tf_key not in VALID_TFS: return pd.DataFrame()
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, tf_key, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = df['time'] // 1000
        df.ta.rsi(length=14, append=True)
        df.rename(columns={'RSI_14': 'rsi'}, inplace=True)
        df = df[['time', 'open', 'high', 'low', 'close', 'rsi']].dropna()
        return df
    except Exception as e:
        return pd.DataFrame()

def last_price():
    try: return float(exchange.fetch_ticker(SYMBOL)["last"])
    except: return 0.0

def pack_df_to_list(df): 
    if df.empty: return [] 
    return df[["time","open","high","low","close","rsi"]].to_dict("records")

def calculate_pnl():
    pnl = 0.0
    current_price = last_price()
    if not current_price: return 0.0
    
    for t in STATE["openTrades"]:
        entry = t['entryPrice']
        size = t['size']
        fee_rate = t.get('fee_rate', GENERAL_CONFIG['fee']) / 100.0
        
        # Calculate total fee cost (entry + exit)
        total_fee_cost = (entry * size * fee_rate) + (current_price * size * fee_rate)
        
        if t['side'] == 'LONG':
            unrealized_pnl = (current_price - entry) * size - total_fee_cost
        else:
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
    if op == ">": return (prev1 <= prev2) and (curr1 > curr2)
    if op == "<": return (prev1 >= prev2) and (curr1 < curr2)
    return False

def open_trade(side, price, logic="Manual"):
    t = {
        "id": str(uuid.uuid4())[:8], "side": side, "size": float(GENERAL_CONFIG["qty"]),
        "entryPrice": float(price), 
        "sl": GENERAL_CONFIG.get("sl"), "tp": GENERAL_CONFIG.get("tp"),
        "fee_rate": float(GENERAL_CONFIG["fee"]), "pnl": 0.0, "auto": True, "logic": logic,
        "time": datetime.now().isoformat() 
    }
    STATE["openTrades"].append(t)
    
    if trades_collection is not None: 
        try: 
            trades_collection.insert_one(t.copy())
        except: 
            pass
    update_db()

def close_trade(t, price, reason):
    if t in STATE["openTrades"]:
        entry = t['entryPrice']
        size = t['size']
        fee_rate = t.get('fee_rate', GENERAL_CONFIG['fee']) / 100.0
        total_fee_cost = (entry * size * fee_rate) + (price * size * fee_rate)
        
        if t['side'] == 'LONG': realized_pnl = (price - entry) * size - total_fee_cost
        else: realized_pnl = (entry - price) * size - total_fee_cost

        STATE["wallet"] += realized_pnl
        h = {
            "time": datetime.now().isoformat(), "side": t["side"],
            "entryPrice": t["entryPrice"], "exitPrice": float(price), "qty": t["size"],
            "realizedPnl": realized_pnl, "logic": t.get("logic", "N/A"), "reason": reason
        }
        STATE["history"].append(h)
        STATE["openTrades"].remove(t)
        
        # --- FIXED DB UPDATES ---
        if trades_collection is not None: 
            try: 
                trades_collection.delete_one({"id": t["id"]})
            except: 
                pass
        
        if history_collection is not None: 
            try: 
                history_collection.insert_one(h.copy())
            except: 
                pass
        
        update_db()

# --- RSI STRATEGY LOGIC ---
def check_rsi_signal(config, df_stf_f, df_stf_s, df_ltf_f, df_ltf_s, side):
    min_len = 2
    if any(len(df) < min_len for df in [df_stf_f, df_stf_s, df_ltf_f, df_ltf_s]): return False, None

    try:
        stf_f, stf_s = df_stf_f["rsi"].iloc[-1], df_stf_s["rsi"].iloc[-1]
        stf_f_prev, stf_s_prev = df_stf_f["rsi"].iloc[-2], df_stf_s["rsi"].iloc[-2]
        ltf_f, ltf_s = df_ltf_f["rsi"].iloc[-1], df_ltf_s["rsi"].iloc[-1]
        ltf_f_prev, ltf_s_prev = df_ltf_f["rsi"].iloc[-2], df_ltf_s["rsi"].iloc[-2]
        
        if math.isnan(stf_f) or math.isnan(ltf_f): return False, None
        
        stf_op, ltf_op = config["stf_logic"], config["ltf_logic"]
        entry_mode = config["entry_mode"]
        
        stf_cross = check_cross(stf_f_prev, stf_f, stf_op, stf_s_prev, stf_s)
        ltf_filter = check_condition(ltf_f, ltf_op, ltf_s)
        ltf_cross = check_cross(ltf_f_prev, ltf_f, ltf_op, ltf_s_prev, ltf_s)

        trade_allowed = False
        if entry_mode == "BOTH": trade_allowed = stf_cross and ltf_filter
        elif entry_mode == "STF_ONLY": trade_allowed = stf_cross
        elif entry_mode == "LTF_ONLY": trade_allowed = ltf_cross

        if trade_allowed:
            logic_str = f"RSI {side} | M:{entry_mode}"
            return True, logic_str
        return False, None
    except Exception as e:
        print(f"RSI Check Error: {e}")
        return False, None

# --- NEW: LIQUIDITY SWEEP LOGIC ---
def find_pivots(df, length):
    """Identifies High and Low pivots in the dataframe."""
    pivots = []
    # Iterate through candles, respecting the length lookback/lookforward
    # Note: A pivot at index 'i' is only confirmed at index 'i + length'
    # We scan up to len(df) - length
    if len(df) < length * 2 + 1: return []

    for i in range(length, len(df) - length):
        # Check High
        current_high = df['high'].iloc[i]
        is_high = True
        # Check left and right neighbors
        for j in range(1, length + 1):
            if df['high'].iloc[i-j] > current_high or df['high'].iloc[i+j] > current_high:
                is_high = False; break
        if is_high: pivots.append({'type': 'HIGH', 'price': current_high, 'index': i})

        # Check Low
        current_low = df['low'].iloc[i]
        is_low = True
        for j in range(1, length + 1):
            if df['low'].iloc[i-j] < current_low or df['low'].iloc[i+j] < current_low:
                is_low = False; break
        if is_low: pivots.append({'type': 'LOW', 'price': current_low, 'index': i})
        
    return pivots

def check_liquidity_signal(df, config):
    """
    Checks if the *current* candle has swept a valid, unbroken liquidity level.
    """
    if not config.get("enabled", False): return None, None, None

    # Determine which structures to check
    structures_to_check = [("EXT", config["ext_len"])]
    if config.get("trade_int", False):
        structures_to_check.append(("INT", config["int_len"]))

    current_price_obj = df.iloc[-1] # Current live candle
    curr_high = current_price_obj['high']
    curr_low = current_price_obj['low']
    
    # Iterate over EXT then INT
    for struct_name, length in structures_to_check:
        pivots = find_pivots(df[:-1], length) # Exclude current candle from forming a pivot
        
        # Check Highs (Liquidity at Top -> Sweep -> Short)
        highs = [p for p in pivots if p['type'] == 'HIGH']
        for h in highs:
            # Is this line still active? (Has price broken it between formation and now?)
            # Look from pivot index + 1 up to current candle index - 1
            broken = False
            subset = df['high'].iloc[h['index']+1 : -1] # Highs between pivot and now
            if not subset.empty and subset.max() > h['price']: broken = True
            
            if not broken:
                # Check if CURRENT candle broke it
                if curr_high > h['price']:
                    # SWEEP DETECTED! Reversal Short
                    return "SHORT", f"Liq Sweep {struct_name} High", h['price']

        # Check Lows (Liquidity at Bottom -> Sweep -> Long)
        lows = [p for p in pivots if p['type'] == 'LOW']
        for l in lows:
            broken = False
            subset = df['low'].iloc[l['index']+1 : -1]
            if not subset.empty and subset.min() < l['price']: broken = True
            
            if not broken:
                if curr_low < l['price']:
                    # SWEEP DETECTED! Reversal Long
                    return "LONG", f"Liq Sweep {struct_name} Low", l['price']
                    
    return None, None, None


# --- BOT LOOP ---
def bot_loop():
    while True:
        try:
            current_price = last_price()
            if not current_price:
                time.sleep(3); continue

            calculate_pnl()
            check_exit_signal(current_price)
            
            if STATE["running"]:
                # Prevent conflicting auto trades
                if not any(t.get('auto') for t in STATE["openTrades"]):
                    
                    # 1. Fetch Data
                    # We combine RSI TFs + Liquidity TFs (default 5m for Liq)
                    req_tfs = set([
                        BUY_CONFIG["stf1"], BUY_CONFIG["stf2"], BUY_CONFIG["ltf1"], BUY_CONFIG["ltf2"],
                        SELL_CONFIG["stf1"], SELL_CONFIG["stf2"], SELL_CONFIG["ltf1"], SELL_CONFIG["ltf2"],
                        LIQ_CONFIG.get("tf", "5m")
                    ])
                    dfs = {tf: fetch_candles(tf, limit=300) for tf in req_tfs}
                    
                    signal_found = False

                    # 2. CHECK LIQUIDITY STRATEGY (Priority)
                    if LIQ_CONFIG.get("enabled"):
                        liq_df = dfs.get(LIQ_CONFIG.get("tf", "5m"))
                        if liq_df is not None and not liq_df.empty:
                            liq_side, liq_logic, liq_level = check_liquidity_signal(liq_df, LIQ_CONFIG)
                            if liq_side:
                                open_trade(liq_side, current_price, liq_logic)
                                signal_found = True

                    # 3. CHECK RSI STRATEGY (Only if no Liquidity signal)
                    if not signal_found:
                        # Check Buy
                        buy_trig, buy_log = check_rsi_signal(
                            BUY_CONFIG, 
                            dfs.get(BUY_CONFIG["stf1"]), dfs.get(BUY_CONFIG["stf2"]),
                            dfs.get(BUY_CONFIG["ltf1"]), dfs.get(BUY_CONFIG["ltf2"]), "LONG"
                        )
                        # Check Sell
                        sell_trig, sell_log = check_rsi_signal(
                            SELL_CONFIG, 
                            dfs.get(SELL_CONFIG["stf1"]), dfs.get(SELL_CONFIG["stf2"]),
                            dfs.get(SELL_CONFIG["ltf1"]), dfs.get(SELL_CONFIG["ltf2"]), "SHORT"
                        )

                        if buy_trig and not sell_trig: open_trade("LONG", current_price, buy_log)
                        elif sell_trig and not buy_trig: open_trade("SHORT", current_price, sell_log)

            update_db()
            time.sleep(5)

        except Exception as e:
            print(f"Bot Loop Error: {e}")
            time.sleep(5)

def check_exit_signal(current_price):
    trades_to_close = []
    for t in STATE["openTrades"]:
        exit_reason = None
        sl = t.get('sl') if t.get('sl') and t.get('sl') > 0 else GENERAL_CONFIG.get('sl')
        tp = t.get('tp') if t.get('tp') and t.get('tp') > 0 else GENERAL_CONFIG.get('tp')
        
        if sl and sl > 0:
            if t['side'] == 'LONG' and current_price <= sl: exit_reason = "SL Hit"
            elif t['side'] == 'SHORT' and current_price >= sl: exit_reason = "SL Hit"
        if not exit_reason and tp and tp > 0:
            if t['side'] == 'LONG' and current_price >= tp: exit_reason = "TP Hit"
            elif t['side'] == 'SHORT' and current_price <= tp: exit_reason = "TP Hit"
        
        if exit_reason: trades_to_close.append((t, exit_reason))

    for t, reason in trades_to_close: close_trade(t, current_price, reason)


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
        
        saved = state_collection.find_one({"_id": "global_state"})
        if saved:
            STATE["wallet"] = saved.get("wallet", 1000.0)
            STATE["running"] = saved.get("running", False)
            if "buy_config" in saved: BUY_CONFIG.update(saved["buy_config"])
            if "sell_config" in saved: SELL_CONFIG.update(saved["sell_config"])
            if "general_config" in saved: GENERAL_CONFIG.update(saved["general_config"])
            if "liq_config" in saved: LIQ_CONFIG.update(saved["liq_config"]) # Load Liquidity Config
        else:
            state_collection.insert_one({"_id": "global_state", "wallet": 1000.0, "running": False, "buy_config":

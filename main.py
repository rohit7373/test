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

# --- CORS MIDDLEWARE (CRITICAL for frontend connection) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
exchange = ccxt.binance({"enableRateLimit": True})
SYMBOL = "BTC/USDT"
# NOTE: Set this MONGO_URL environment variable in your Railway dashboard!
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
    "entry_mode": "BOTH", # New: Entry mode (BOTH/STF_ONLY/LTF_ONLY)
    "qty": 0.01,
    "fee": 0.1,  # Fee rate in percent
    "tp": None,  # Take Profit price
    "sl": None,  # Stop Loss price
}

# --- Pydantic Models for Requests ---
class BotConfig(BaseModel):
    stf1: str
    stf2: str
    ltf1: str
    ltf2: str
    stf_logic: str
    ltf_logic: str
    entry_mode: str
    qty: float
    fee: float
    tp: Optional[float] = None
    sl: Optional[float] = None

class ManualOrderReq(BaseModel):
    side: str
    qty: float
    type: str
    sl: Optional[float] = None
    tp: Optional[float] = None

class CloseTradeReq(BaseModel):
    id: str

# --- UTILITIES ---

def clean_data(data):
    """Helper to convert MongoDB ObjectId/datetime to serializable strings."""
    if isinstance(data, list):
        return [clean_data(item) for item in data]
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    if isinstance(data, ObjectId):
        return str(data)
    if isinstance(data, datetime):
        return data.isoformat()
    return data

def last_price():
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        return ticker['last']
    except Exception:
        # Fallback to a fake price if exchange fails
        return 50000.0 

def get_data(timeframe, limit=200):
    """Fetch and prepare OHLCV data with RSI."""
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = df['time'] // 1000  # Convert milliseconds to seconds
        
        # Calculate RSI
        df.ta.rsi(length=14, append=True)
        df.rename(columns={'RSI_14': 'rsi'}, inplace=True)
        
        # Keep only necessary columns and drop rows with NaN (for RSI)
        df = df[['time', 'open', 'high', 'low', 'close', 'rsi']].dropna()
        
        return df.to_dict('records')
    except Exception as e:
        print(f"Error fetching data for {timeframe}: {e}")
        return []

def calculate_rsi(data):
    """Extract only time and RSI value from the OHLCV data."""
    return [{"time": d['time'], "value": d['rsi']} for d in data if 'rsi' in d]

def calculate_pnl():
    """Calculates total unrealized PnL for all open trades."""
    pnl = 0.0
    current_price = last_price()
    if not current_price: return 0.0
    
    for t in STATE["openTrades"]:
        entry = t['entryPrice']
        size = t['size']
        fee = t['fee_rate'] / 100.0 # Convert % to decimal
        
        if t['side'] == 'LONG':
            # Formula: (Current Price - Entry Price) * Size - (Fee on Entry + Fee on Exit)
            unrealized_pnl = (current_price - entry) * size - (entry * size * fee) - (current_price * size * fee)
        else: # SHORT
            # Formula: (Entry Price - Current Price) * Size - (Fee on Entry + Fee on Exit)
            unrealized_pnl = (entry - current_price) * size - (entry * size * fee) - (current_price * size * fee)

        t['pnl'] = unrealized_pnl
        pnl += unrealized_pnl
        
    STATE["unrealized"] = pnl
    return pnl

def check_entry_signal(stf_fast_data, stf_slow_data, ltf_fast_data, ltf_slow_data):
    """
    Checks for RSI crossover signals on STF and uses LTF as a filter/entry depending on mode.
    Returns: ('LONG', 'Logic String') or ('SHORT', 'Logic String') or (None, None)
    """
    if not (stf_fast_data and stf_slow_data and ltf_fast_data and ltf_slow_data):
        return None, None

    # Get the latest RSI values
    stf1_rsi = stf_fast_data[-1].get('rsi')
    stf2_rsi = stf_slow_data[-1].get('rsi')
    ltf1_rsi = ltf_fast_data[-1].get('rsi')
    ltf2_rsi = ltf_slow_data[-1].get('rsi')
    
    # Get the previous RSI values (for crossover detection)
    stf1_rsi_prev = stf_fast_data[-2].get('rsi')
    stf2_rsi_prev = stf_slow_data[-2].get('rsi')
    ltf1_rsi_prev = ltf_fast_data[-2].get('rsi')
    ltf2_rsi_prev = ltf_slow_data[-2].get('rsi')

    if None in [stf1_rsi, stf2_rsi, ltf1_rsi, ltf2_rsi, stf1_rsi_prev, stf2_rsi_prev, ltf1_rsi_prev, ltf2_rsi_prev]:
        return None, None

    logic_op = BOT_CONFIG['stf_logic']
    ltf_op = BOT_CONFIG['ltf_logic']
    
    # Check STF Cross (The primary entry signal)
    stf_long_cross = (stf1_rsi_prev <= stf2_rsi_prev) and (stf1_rsi > stf2_rsi)
    stf_short_cross = (stf1_rsi_prev >= stf2_rsi_prev) and (stf1_rsi < stf2_rsi)

    # Check LTF Filter/Cross (The secondary condition)
    ltf_long_filter = (ltf1_rsi > ltf2_rsi) if ltf_op == '>' else (ltf1_rsi < ltf2_rsi)
    ltf_short_filter = (ltf1_rsi < ltf2_rsi) if ltf_op == '>' else (ltf1_rsi > ltf2_rsi)

    
    if len(STATE["openTrades"]) > 0:
        return None, None # Already in a trade

    if BOT_CONFIG['entry_mode'] == 'STF_ONLY':
        if stf_long_cross and logic_op == '>':
            return 'LONG', f"STF1({stf1_rsi:.2f}) > STF2({stf2_rsi:.2f}) Cross"
        if stf_short_cross and logic_op == '<':
            return 'SHORT', f"STF1({stf1_rsi:.2f}) < STF2({stf2_rsi:.2f}) Cross"
    
    elif BOT_CONFIG['entry_mode'] == 'LTF_ONLY':
        if ltf_long_filter and ltf_short_cross: # Use LTF cross as entry
            return 'LONG', f"LTF1({ltf1_rsi:.2f}) > LTF2({ltf2_rsi:.2f}) Cross"
        if ltf_short_filter and ltf_long_cross: # Use LTF cross as entry
            return 'SHORT', f"LTF1({ltf1_rsi:.2f}) < LTF2({ltf2_rsi:.2f}) Cross"
            
    elif BOT_CONFIG['entry_mode'] == 'BOTH':
        if stf_long_cross and ltf_long_filter and logic_op == '>':
            return 'LONG', f"STF Cross AND LTF Filter ({ltf_op})"
        if stf_short_cross and ltf_short_filter and logic_op == '<':
            return 'SHORT', f"STF Cross AND LTF Filter ({ltf_op})"

    return None, None

def close_trade(t, close_price, logic_reason):
    """
    Processes the closure of a trade, updates PnL, state, and logs history.
    """
    if t not in STATE["openTrades"]: return
    
    entry = t['entryPrice']
    size = t['size']
    fee = BOT_CONFIG['fee'] / 100.0 # Global fee used for all trades

    if t['side'] == 'LONG':
        pnl = (close_price - entry) * size - (entry * size * fee) - (close_price * size * fee)
    else: # SHORT
        pnl = (entry - close_price) * size - (entry * size * fee) - (close_price * size * fee)

    STATE["wallet"] += pnl
    
    # History record
    h = {
        "time": int(time.time()),
        "side": t['side'],
        "size": size,
        "entryPrice": entry,
        "price": close_price,
        "realizedPnl": pnl,
        "logic": logic_reason,
        "trade_id": t['id']
    }
    STATE["history"].append(h)
    
    # Remove from openTrades
    STATE["openTrades"].remove(t)
    
    # Update DB
    if history_collection is not None:
        try: history_collection.insert_one(h.copy())
        except: pass

def check_exit_signal(current_price):
    """Checks for TP/SL signals based on current price."""
    trades_to_close = []
    
    for t in STATE["openTrades"]:
        exit_reason = None
        
        # Check Stop Loss (SL)
        if BOT_CONFIG['sl'] is not None and BOT_CONFIG['sl'] > 0:
            if t['side'] == 'LONG' and current_price <= BOT_CONFIG['sl']:
                exit_reason = "SL Hit"
            elif t['side'] == 'SHORT' and current_price >= BOT_CONFIG['sl']:
                exit_reason = "SL Hit"

        # Check Take Profit (TP) - SL takes priority
        if exit_reason is None and BOT_CONFIG['tp'] is not None and BOT_CONFIG['tp'] > 0:
            if t['side'] == 'LONG' and current_price >= BOT_CONFIG['tp']:
                exit_reason = "TP Hit"
            elif t['side'] == 'SHORT' and current_price <= BOT_CONFIG['tp']:
                exit_reason = "TP Hit"
        
        if exit_reason:
            trades_to_close.append((t, exit_reason))

    # Execute closures
    for t, reason in trades_to_close:
        close_trade(t, current_price, reason)

def run_bot():
    """Main bot loop executed in a separate thread."""
    while True:
        try:
            if STATE["running"]:
                current_price = last_price()
                if not current_price:
                    time.sleep(3)
                    continue

                # 1. Fetch data
                stf1_data = get_data(BOT_CONFIG['stf1'], limit=30)
                stf2_data = get_data(BOT_CONFIG['stf2'], limit=30)
                ltf1_data = get_data(BOT_CONFIG['ltf1'], limit=30)
                ltf2_data = get_data(BOT_CONFIG['ltf2'], limit=30)

                # 2. Check Exit Signals (TP/SL)
                check_exit_signal(current_price)
                
                # 3. Check Entry Signals (Only if no open trades)
                if not STATE["openTrades"]:
                    side, logic = check_entry_signal(stf1_data, stf2_data, ltf1_data, ltf2_data)

                    if side:
                        entry_price = current_price # Execute trade at current market price
                        
                        t = {
                            "id": str(uuid.uuid4())[:8],
                            "side": side,
                            "size": BOT_CONFIG['qty'],
                            "entryPrice": entry_price,
                            "sl": BOT_CONFIG['sl'],
                            "tp": BOT_CONFIG['tp'],
                            "fee_rate": BOT_CONFIG['fee'],
                            "pnl": 0.0,
                            "logic": logic,
                            "time": datetime.now().isoformat()
                        }
                        STATE["openTrades"].append(t)
                        print(f"BOT ENTRY: {side} @ {entry_price:.2f} with Logic: {logic}")
                        
                        # Store trade to DB
                        if trades_collection is not None:
                            try: trades_collection.insert_one(t.copy())
                            except: pass

                # 4. Update Unrealized PnL (for display)
                calculate_pnl()
                
                # 5. Save State
                if state_collection is not None:
                    try: state_collection.update_one(
                        {"_id": "bot_state"}, 
                        {"$set": {"wallet": STATE["wallet"], "running": STATE["running"]}}, 
                        upsert=True
                    )
                    except: pass
                
            else:
                # Still calculate PnL even when stopped if there are open trades
                calculate_pnl()
                
            time.sleep(3) # Wait 3 seconds before next iteration

        except Exception as e:
            print(f"Bot Error: {e}")
            time.sleep(5)

# --- FastAPI LIFECYCLE HOOKS ---
@app.on_event("startup")
def startup_event():
    """Connects to MongoDB and starts the bot thread."""
    global db, state_collection, trades_collection, history_collection
    
    if MONGO_URL:
        try:
            client = MongoClient(MONGO_URL)
            db = client.rsi_bot_db
            state_collection = db.state
            trades_collection = db.trades
            history_collection = db.history
            
            # Load initial state from DB
            loaded_state = state_collection.find_one({"_id": "bot_state"})
            if loaded_state and loaded_state.get("wallet") is not None:
                STATE["wallet"] = loaded_state["wallet"]
            if loaded_state and loaded_state.get("running") is not None:
                STATE["running"] = loaded_state["running"]
            
            # Load open trades (optional, can be empty on startup)
            STATE["openTrades"] = list(trades_collection.find({"closed_time": {"$exists": False}}))
            
            # Load history (last 20 records)
            STATE["history"] = list(history_collection.find().sort("time", -1).limit(20))
            
            print("Successfully connected to MongoDB and loaded state.")
            
        except Exception as e:
            print(f"MongoDB connection failed: {e}. Running without persistence.")
            state_collection = None # Set to None to disable DB operations

    # Start bot thread regardless of DB status
    threading.Thread(target=run_bot, daemon=True).start()


# --- API ENDPOINTS ---

@app.get("/api/market")
async def get_market_data():
    """Fetches and returns all market data, state, and config for the frontend."""
    # Fetch data concurrently (last 30 bars should be enough for charts)
    stf1_data = get_data(BOT_CONFIG['stf1'], limit=30)
    stf2_data = get_data(BOT_CONFIG['stf2'], limit=30)
    ltf1_data = get_data(BOT_CONFIG['ltf1'], limit=30)
    ltf2_data = get_data(BOT_CONFIG['ltf2'], limit=30)

    # Calculate RSI lists (Frontend expects this format for the RSI charts)
    stf1_rsi_list = calculate_rsi(stf1_data)
    stf2_rsi_list = calculate_rsi(stf2_data)
    ltf1_rsi_list = calculate_rsi(ltf1_data)
    ltf2_rsi_list = calculate_rsi(ltf2_data)
    
    response_data = {
        "price": last_price(),
        "config": BOT_CONFIG,
        "state": {
            "running": STATE["running"],
            "wallet": STATE["wallet"],
            "unrealized": STATE["unrealized"],
        },
        # OHLCV data for candlestick charts
        "stf1": stf1_data,
        "stf2": stf2_data,
        "ltf1": ltf1_data,
        "ltf2": ltf2_data,
        # RSI data for RSI charts (Frontend will use the OHLCV data, but providing RSI lists too)
        "stf1_rsi": stf1_rsi_list,
        "stf2_rsi": stf2_rsi_list,
        "ltf1_rsi": ltf1_rsi_list,
        "ltf2_rsi": ltf2_rsi_list,
        # Trade tables
        "openTrades": clean_data(STATE["openTrades"]),
        "history": clean_data(STATE["history"]),
    }
    
    # NOTE: The frontend's mapRsi and normalizeCandles handles parsing the single 'stf1'/'stf2' array.
    
    return JSONResponse(content=response_data)


@app.post("/api/start")
def start_bot(config: BotConfig):
    """Starts the bot and updates configuration."""
    if STATE["running"]:
        raise HTTPException(status_code=400, detail="Bot is already running")
    
    # Update config
    for k, v in config.dict().items():
        BOT_CONFIG[k] = v
        
    STATE["running"] = True
    
    # Save running state to DB
    if state_collection is not None:
        try: state_collection.update_one(
            {"_id": "bot_state"}, 
            {"$set": {"running": True}}, 
            upsert=True
        )
        except: pass
        
    return {"status": "success", "message": "Bot started"}

@app.post("/api/stop")
def stop_bot():
    """Stops the bot."""
    if not STATE["running"]:
        raise HTTPException(status_code=400, detail="Bot is already stopped")
        
    STATE["running"] = False
    
    # Save running state to DB
    if state_collection is not None:
        try: state_collection.update_one(
            {"_id": "bot_state"}, 
            {"$set": {"running": False}}, 
            upsert=True
        )
        except: pass
        
    return {"status": "success", "message": "Bot stopped"}


@app.post("/api/manual/order")
def manual_order(order: ManualOrderReq):
    """Places a manual trade (LONG or SHORT)."""
    try:
        price = last_price()
        if not price:
             raise HTTPException(status_code=500, detail="Could not fetch current price.")
             
        t = {
            "id": str(uuid.uuid4())[:8],
            "side": order.side, 
            "size": float(order.qty),
            "entryPrice": price, 
            "sl": order.sl, 
            "tp": order.tp, 
            "fee_rate": BOT_CONFIG['fee'],
            "pnl": 0.0, 
            "logic": "Manual", 
            "time": datetime.now().timestamp() # Use timestamp for consistency
        }
        STATE["openTrades"].append(t)
        
        # FIX: Explicit check and DB insertion
        if trades_collection is not None: 
            try: trades_collection.insert_one(t.copy())
            except: pass
            
        return clean_data({"status": "success", "trade": t})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/manual/close")
def manual_close(req: CloseTradeReq):
    """Closes a single specific trade by ID."""
    try:
        t = next((x for x in STATE["openTrades"] if x["id"] == req.id), None)
        if t:
            close_trade(t, last_price(), "Manual Close")
            # NOTE: DB update happens inside close_trade for history.
            return {"status": "success"}
        raise HTTPException(status_code=404, detail="Trade not found")
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/manual/close-all")
def close_all():
    """Closes all open trades."""
    try:
        p = last_price()
        count = len(STATE["openTrades"])
        
        # Iterate over a copy because close_trade modifies the original list
        for t in list(STATE["openTrades"]): 
            close_trade(t, p, "Close All")
            
        return {"status": "success", "closed_count": count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# Use `if __name__ == "__main__":` to run locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

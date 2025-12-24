import os
import threading
import time
import pandas as pd
import pandas_ta as ta
import ccxt
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# --- APP SETUP ---
app = FastAPI()

# Allow CORS so your HTML frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- EXCHANGE SETUP ---
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'} 
})

# --- CONFIGURATION & STATE ---
BOT_CONFIG = {
    "symbol": "BTC/USDT",
    "htf_fast": "1h",
    "htf_slow": "4h",
    "ltf_fast": "5m",
    "ltf_slow": "15m",
    "bot_qty": 0.001,
    "strategy_mode": "BOTH"
}

BOT_STATE = {
    "running": False,
    "htf_signal": "WAIT",
    "wallet_balance": 1000.0,
    "position": None,      # None, "LONG", or "SHORT"
    "entry_price": 0.0,
    "pos_size": 0.0,
    "trades": []
}

# --- HELPER FUNCTIONS ---
def get_rsi_value(symbol, timeframe, length=14):
    """Fetches candles and calculates the latest RSI value."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        if not ohlcv: return 50
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        df['rsi'] = ta.rsi(df['close'], length=length)
        if pd.isna(df['rsi'].iloc[-1]): return 50
        return float(df['rsi'].iloc[-1])
    except Exception as e:
        print(f"[Data Error] {timeframe}: {e}")
        return 50

def get_current_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return float(ticker['last'])
    except:
        return 0.0

def execute_trade(side, order_type, qty, price, reason="BOT"):
    timestamp = datetime.now().strftime("%H:%M:%S")

    # 1. Close Opposite Position if exists
    if BOT_STATE["position"]:
        if (BOT_STATE["position"] == "LONG" and side == "SELL") or \
           (BOT_STATE["position"] == "SHORT" and side == "BUY"):
            close_position(price)

    # 2. Open New Position
    if BOT_STATE["position"] is None:
        BOT_STATE["position"] = "LONG" if side == "BUY" else "SHORT"
        BOT_STATE["entry_price"] = price
        BOT_STATE["pos_size"] = qty
        
        log_entry = {
            "time": timestamp,
            "side": side,
            "type": reason,
            "price": price,
            "pnl": 0.0
        }
        BOT_STATE["trades"].insert(0, log_entry)
        print(f"[{timestamp}] OPEN {side} @ {price}")

def close_position(price):
    if not BOT_STATE["position"]: return

    qty = BOT_STATE["pos_size"]
    entry = BOT_STATE["entry_price"]
    is_long = BOT_STATE["position"] == "LONG"
    
    # Calculate PnL
    diff = price - entry
    if not is_long: diff = -diff
    pnl = diff * qty

    BOT_STATE["wallet_balance"] += pnl
    
    log_entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "side": "SELL" if is_long else "BUY",
        "type": "CLOSE",
        "price": price,
        "pnl": round(pnl, 2)
    }
    BOT_STATE["trades"].insert(0, log_entry)
    
    BOT_STATE["position"] = None
    BOT_STATE["entry_price"] = 0.0
    BOT_STATE["pos_size"] = 0.0
    print(f"Closed Position. PnL: {pnl:.2f}")

# --- BOT LOGIC LOOP ---
def bot_logic_loop():
    print("--- Bot Loop Thread Started ---")
    while True:
        if BOT_STATE["running"]:
            try:
                sym = BOT_CONFIG["symbol"]
                price = get_current_price(sym)
                
                # Fetch RSI values
                rh_f = get_rsi_value(sym, BOT_CONFIG["htf_fast"])
                rh_s = get_rsi_value(sym, BOT_CONFIG["htf_slow"])
                rl_f = get_rsi_value(sym, BOT_CONFIG["ltf_fast"])
                rl_s = get_rsi_value(sym, BOT_CONFIG["ltf_slow"])
                
                # Determine Signals
                htf_bull = rh_f > rh_s
                ltf_bull = rl_f > rl_s
                
                BOT_STATE["htf_signal"] = "UP (Bullish)" if htf_bull else "DOWN (Bearish)"

                # Execution Logic
                # BUY: HTF Bull + LTF Bull + No Position
                if htf_bull and ltf_bull:
                    if BOT_STATE["position"] is None:
                        execute_trade("BUY", "MARKET", BOT_CONFIG["bot_qty"], price, "SIGNAL")
                    elif BOT_STATE["position"] == "SHORT":
                        close_position(price)

                # SELL: HTF Bear + LTF Bear + No Position
                elif not htf_bull and not ltf_bull:
                    if BOT_STATE["position"] is None:
                        execute_trade("SELL", "MARKET", BOT_CONFIG["bot_qty"], price, "SIGNAL")
                    elif BOT_STATE["position"] == "LONG":
                        close_position(price)

            except Exception as e:
                print(f"Loop Logic Error: {e}")
        
        time.sleep(2) # Check every 2 seconds

# Start the background thread
t = threading.Thread(target=bot_logic_loop, daemon=True)
t.start()

# --- API ENDPOINTS ---
@app.get("/")
def index():
    return {"status": "Bot Backend Running"}

@app.get("/bot/status")
def get_status():
    return {
        "config": BOT_CONFIG,
        "state": {
            "running": BOT_STATE["running"],
            "htf_signal": BOT_STATE["htf_signal"],
            "position": BOT_STATE["position"],
            "wallet": BOT_STATE["wallet_balance"],
            "trades": BOT_STATE["trades"][:50]
        }
    }

@app.post("/bot/start")
def start_bot():
    BOT_STATE["running"] = True
    return {"status": "started"}

@app.post("/bot/stop")
def stop_bot():
    BOT_STATE["running"] = False
    return {"status": "stopped"}

@app.post("/bot/config")
async def update_config(req: Request):
    data = await req.json()
    BOT_CONFIG.update(data)
    return {"status": "updated", "config": BOT_CONFIG}

@app.post("/bot/manual")
async def manual_trade(req: Request):
    data = await req.json()
    side = data.get("side")
    qty = float(data.get("qty", 0.001))
    price = float(data.get("price", 0) or get_current_price(BOT_CONFIG["symbol"]))
    execute_trade(side, "MARKET", qty, price, "MANUAL")
    return {"status": "executed"}

if __name__ == "__main__":
    # RAILWAY CONFIGURATION
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

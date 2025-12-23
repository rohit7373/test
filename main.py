import os, time, math, threading, datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client

app = FastAPI()

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
# Use environment variables or hardcode for testing. 
# NOTE: Use Testnet keys for safety during development.
API_KEY = os.getenv("BINANCE_API_KEY", "YOUR_TESTNET_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY", "YOUR_TESTNET_SECRET")

client = Client(API_KEY, API_SECRET, testnet=True)
# Ensure we point to testnet URL if using testnet keys
client.API_URL = "https://testnet.binance.vision/api"

BOT_CONFIG = {
    "symbol": "BTCUSDT",
    "htf_fast": "1h", 
    "htf_slow": "4h",
    "ltf_fast": "5m", 
    "ltf_slow": "15m",
    "rsi_len": 14,
    "max_gap_mins": 120,
    "pos_pct": 10,
    "enabled": False
}

BOT_STATE = {
    "running": False,
    "htf_signal": "WAIT", 
    "htf_time": 0,
    "position": None, 
    "trades": [],
    "last_check": "Never"
}

# --- LOGIC ---
def get_rsi(symbol, tf, length):
    """Calculates RSI using Binance Klines"""
    try:
        # Fetch enough candles to calculate RSI accurately
        kl = client.get_klines(symbol=symbol, interval=tf, limit=length + 50)
        closes = [float(k[4]) for k in kl]
        
        if len(closes) < length: return 50
        
        # Calculate RSI
        alpha = 1 / length
        gains = []
        losses = []

        # Initial SMA
        for i in range(1, length + 1):
            change = closes[i] - closes[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
            
        avg_gain = sum(gains) / length
        avg_loss = sum(losses) / length
        
        # RMA (Wilder's Smoothing)
        for i in range(length + 1, len(closes)):
            change = closes[i] - closes[i - 1]
            gain = max(change, 0)
            loss = max(-change, 0)
            avg_gain = (avg_gain * (length - 1) + gain) / length
            avg_loss = (avg_loss * (length - 1) + loss) / length

        if avg_loss == 0: return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        print(f"RSI Error: {e}")
        return 50

def execute_trade(side, order_type):
    """Simulates or executes a trade"""
    try:
        # Fetch current price for simulation
        ticker = client.get_symbol_ticker(symbol=BOT_CONFIG["symbol"])
        curr_price = float(ticker["price"])
        
        # Log trade
        trade = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "timestamp": time.time(),
            "side": side,
            "price": curr_price,
            "type": order_type
        }
        BOT_STATE["trades"].append(trade)
        
        # Update Position State
        if side == "BUY":
            BOT_STATE["position"] = "LONG"
        elif side == "SELL":
            BOT_STATE["position"] = None # Flat
            
        print(f"Executed {side} at {curr_price}")
        return True
    except Exception as e:
        print(f"Trade Error: {e}")
        return False

def bot_loop():
    """Main trading loop running in a separate thread"""
    BOT_STATE["running"] = True
    print("Bot Loop Started...")
    
    while BOT_CONFIG["enabled"]:
        try:
            now = time.time()
            sym = BOT_CONFIG["symbol"]
            
            # 1. Calculate HTF RSI
            h_f = get_rsi(sym, BOT_CONFIG["htf_fast"], BOT_CONFIG["rsi_len"])
            h_s = get_rsi(sym, BOT_CONFIG["htf_slow"], BOT_CONFIG["rsi_len"])
            
            # Determine Bias
            if h_f > h_s: 
                BOT_STATE["htf_signal"] = "UP (Bullish)"
                BOT_STATE["htf_time"] = now # Mark the time bias became valid
            elif h_f < h_s: 
                BOT_STATE["htf_signal"] = "DOWN (Bearish)"
                BOT_STATE["htf_time"] = now

            # 2. Calculate LTF RSI
            l_f = get_rsi(sym, BOT_CONFIG["ltf_fast"], BOT_CONFIG["rsi_len"])
            l_s = get_rsi(sym, BOT_CONFIG["ltf_slow"], BOT_CONFIG["rsi_len"])
            
            # 3. Logic Check
            # BUY Logic: Bias is Up, No Position, LTF Fast crosses above LTF Slow
            if BOT_STATE["position"] is None and "UP" in BOT_STATE["htf_signal"]:
                if l_f > l_s:
                    execute_trade("BUY", "MARKET")
            
            # SELL Logic: Have Position, LTF Fast crosses below LTF Slow
            elif BOT_STATE["position"] == "LONG":
                if l_f < l_s:
                    execute_trade("SELL", "MARKET")

            BOT_STATE["last_check"] = datetime.datetime.now().strftime("%H:%M:%S")
            
        except Exception as e:
            print(f"Loop Error: {e}")
            
        time.sleep(10) # Wait 10 seconds between checks

    BOT_STATE["running"] = False
    print("Bot Loop Stopped.")

# --- ENDPOINTS ---

@app.post("/bot/start")
def start_bot():
    if not BOT_STATE["running"]:
        BOT_CONFIG["enabled"] = True
        threading.Thread(target=bot_loop, daemon=True).start()
    return {"status": "started"}

@app.post("/bot/stop")
def stop_bot():
    BOT_CONFIG["enabled"] = False
    return {"status": "stopping"}

@app.post("/bot/close_all")
def close_all():
    if BOT_STATE["position"] == "LONG":
        execute_trade("SELL", "MARKET")
    BOT_STATE["trades"] = [] # Reset history for dashboard
    return {"status": "closed_and_reset"}

@app.get("/bot/status")
def get_status():
    return {"config": BOT_CONFIG, "state": BOT_STATE}

@app.post("/bot/config")
async def update_config(req: Request):
    if BOT_STATE["running"]: 
        return {"error": "Stop bot before editing config"}
    data = await req.json()
    BOT_CONFIG.update(data)
    return BOT_CONFIG

@app.post("/bot/manual")
async def manual_trade(req: Request):
    data = await req.json()
    side = data.get("side")
    execute_trade(side, "MANUAL")
    return {"status": "executed"}

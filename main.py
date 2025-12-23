import os, time, math, threading, datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize Binance Client (Testnet)
# Replace environment variables or hardcode for testing
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY")
client = Client(API_KEY, API_SECRET, testnet=True)
client.API_URL = "https://testnet.binance.vision/api"

# Configuration
CONFIG = {
    "symbol": "BTCUSDT",
    "htf_fast": "1h", "htf_slow": "4h",
    "ltf_fast": "5m", "ltf_slow": "15m",
    "rsi_len": 14,
    "max_gap_mins": 120,
    "pos_pct": 10,
    "enabled": False
}

# Bot State
STATE = {
    "running": False,
    "htf_signal": None, # "BUY", "SELL", or None
    "htf_time": 0,
    "position": None, 
    "trades": [],
    "last_check": ""
}

def get_rsi(symbol, tf, length):
    kl = client.get_klines(symbol=symbol, interval=tf, limit=length + 10)
    closes = [float(k[4]) for k in kl]
    g, l = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        g.append(max(d, 0)); l.append(max(-d, 0))
    ag = sum(g[-length:]) / length
    al = sum(l[-length:]) / length if sum(l[-length:]) else 1e-9
    return 100 - (100 / (1 + ag / al))

def bot_loop():
    STATE["running"] = True
    while CONFIG["enabled"]:
        try:
            now = time.time()
            # 1. Check HTF (Yellow vs Red)
            h_f = get_rsi(CONFIG["symbol"], CONFIG["htf_fast"], CONFIG["rsi_len"])
            h_s = get_rsi(CONFIG["symbol"], CONFIG["htf_slow"], CONFIG["rsi_len"])
            
            # Detect HTF Crosses
            if h_f > h_s and STATE["htf_signal"] != "BUY":
                STATE["htf_signal"] = "BUY"
                STATE["htf_time"] = now
            elif h_f < h_s and STATE["htf_signal"] != "SELL":
                STATE["htf_signal"] = "SELL"
                STATE["htf_time"] = now

            # 2. Check LTF (White vs Orange)
            l_f = get_rsi(CONFIG["symbol"], CONFIG["ltf_fast"], CONFIG["rsi_len"])
            l_s = get_rsi(CONFIG["symbol"], CONFIG["ltf_slow"], CONFIG["rsi_len"])
            
            mins_since_htf = (now - STATE["htf_time"]) / 60
            
            # Logic: HTF must match LTF within the time window
            if STATE["position"] is None and STATE["htf_signal"] == "BUY":
                if l_f > l_s and mins_since_htf <= CONFIG["max_gap_mins"]:
                    execute_trade("BUY")
            
            elif STATE["position"] == "BUY" and STATE["htf_signal"] == "SELL":
                if l_f < l_s:
                    execute_trade("SELL")

            STATE["last_check"] = datetime.datetime.now().strftime("%H:%M:%S")
        except Exception as e:
            print(f"Bot Error: {e}")
        
        time.sleep(30) # Check frequency
    STATE["running"] = False

def execute_trade(side):
    price = float(client.get_symbol_ticker(symbol=CONFIG["symbol"])["price"])
    trade = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.time(),
        "side": side,
        "price": price,
        "desc": f"HTF: {STATE['htf_signal']} | LTF: {side}"
    }
    STATE["trades"].append(trade)
    STATE["position"] = "BUY" if side == "BUY" else None
    print(f"Executed {side} at {price}")

@app.post("/bot/start")
def start():
    if not STATE["running"]:
        CONFIG["enabled"] = True
        threading.Thread(target=bot_loop, daemon=True).start()
        return {"status": "started"}
    return {"status": "already_running"}

@app.post("/bot/stop")
def stop():
    CONFIG["enabled"] = False
    return {"status": "stopped"}

@app.get("/bot/status")
def status():
    return {"config": CONFIG, "state": STATE}

@app.post("/bot/config")
async def update_config(req: Request):
    if STATE["running"]:
        return {"error": "Bot is currently running! Please STOP the bot before changing settings."}
    new_cfg = await req.json()
    CONFIG.update(new_cfg)
    return CONFIG

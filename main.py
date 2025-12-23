import os, time, math, threading, datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET_KEY"), testnet=True)
client.API_URL = "https://testnet.binance.vision/api"

BOT_CONFIG = {
    "symbol": "BTCUSDT",
    "htf_fast": "1h", "htf_slow": "4h",
    "ltf_fast": "5m", "ltf_slow": "15m",
    "rsi_len": 14,
    "max_gap_mins": 120,
    "enabled": False
}

# Added htf_crosses to track the exact moments for chart markers
BOT_STATE = {
    "running": False, 
    "htf_signal": None, 
    "htf_time": 0, 
    "position": None, 
    "trades": [],
    "htf_crosses": [] 
}

def get_rsi(symbol, tf, length):
    try:
        kl = client.get_klines(symbol=symbol, interval=tf, limit=length + 50)
        closes = [float(k[4]) for k in kl]
        g, l = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i-1]
            g.append(max(d, 0)); l.append(max(-d, 0))
        ag = sum(g[-length:]) / length
        al = sum(l[-length:]) / length if sum(l[-length:]) else 1e-9
        return 100 - (100 / (1 + ag / al))
    except: return 50

def bot_loop():
    BOT_STATE["running"] = True
    while BOT_CONFIG["enabled"]:
        try:
            now = time.time()
            hf = get_rsi(BOT_CONFIG["symbol"], BOT_CONFIG["htf_fast"], BOT_CONFIG["rsi_len"])
            hs = get_rsi(BOT_CONFIG["symbol"], BOT_CONFIG["htf_slow"], BOT_CONFIG["rsi_len"])
            
            # Detect Yellow crossing Red
            if hf > hs and BOT_STATE["htf_signal"] != "BUY":
                BOT_STATE["htf_signal"], BOT_STATE["htf_time"] = "BUY", now
                BOT_STATE["htf_crosses"].append({"time": now, "side": "UP", "val": hf})
            elif hf < hs and BOT_STATE["htf_signal"] != "SELL":
                BOT_STATE["htf_signal"], BOT_STATE["htf_time"] = "SELL", now
                BOT_STATE["htf_crosses"].append({"time": now, "side": "DOWN", "val": hf})

            lf = get_rsi(BOT_CONFIG["symbol"], BOT_CONFIG["ltf_fast"], BOT_CONFIG["rsi_len"])
            ls = get_rsi(BOT_CONFIG["symbol"], BOT_CONFIG["ltf_slow"], BOT_CONFIG["rsi_len"])
            
            gap = (now - BOT_STATE["htf_time"]) / 60
            if BOT_STATE["position"] is None and BOT_STATE["htf_signal"] == "BUY" and lf > ls and gap <= BOT_CONFIG["max_gap_mins"]:
                BOT_STATE["position"] = "LONG"
                BOT_STATE["trades"].append({"time": now, "side": "BUY", "price": "MARKET"})
            
            elif BOT_STATE["position"] == "LONG" and BOT_STATE["htf_signal"] == "SELL" and lf < ls:
                BOT_STATE["position"] = None
                BOT_STATE["trades"].append({"time": now, "side": "SELL", "price": "MARKET"})
        except: pass
        time.sleep(30)
    BOT_STATE["running"] = False

@app.post("/bot/start")
def start():
    if not BOT_STATE["running"]:
        BOT_CONFIG["enabled"] = True
        threading.Thread(target=bot_loop, daemon=True).start()
    return {"status": "started"}

@app.post("/bot/stop")
def stop():
    BOT_CONFIG["enabled"] = False
    return {"status": "stopped"}

@app.get("/bot/status")
def status():
    return {"config": BOT_CONFIG, "state": BOT_STATE}

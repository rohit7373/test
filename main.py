import os, time, threading, asyncio, json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client
import requests

# ---------------- CONFIG ----------------
BINANCE_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")

TELEGRAM_TOKEN = os.getenv("TG_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID")

client = Client(BINANCE_KEY, BINANCE_SECRET, testnet=True)
client.API_URL = "https://testnet.binance.vision/api"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------- STATE ----------------
BOT_CONFIG = {
    "symbol": "BTCUSDT",
    "rsi_len": 14,
    "tfs": ["1m","5m","15m","1h"],
    "vote_required": 3,
    "enabled": False
}

STATE = {
    "price": 0,
    "rsi": {},
    "position": None,
    "entry": None,
    "pnl": 0,
    "wins": 0,
    "losses": 0,
    "trades": []
}

clients = []

# ---------------- UTIL ----------------
def send_telegram(msg):
    if not TELEGRAM_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

def get_rsi(tf):
    kl = client.get_klines(symbol="BTCUSDT", interval=tf, limit=60)
    closes = [float(k[4]) for k in kl]
    gains, losses = [], []
    for i in range(1,len(closes)):
        d = closes[i]-closes[i-1]
        gains.append(max(d,0))
        losses.append(max(-d,0))
    ag = sum(gains[-14:])/14
    al = sum(losses[-14:])/14 if sum(losses[-14:]) else 1e-9
    return 100 - (100/(1+ag/al))

# ---------------- WS STREAM ----------------
async def price_stream():
    while True:
        p = float(client.get_symbol_ticker(symbol="BTCUSDT")["price"])
        STATE["price"] = p

        votes = 0
        for tf in BOT_CONFIG["tfs"]:
            r = get_rsi(tf)
            STATE["rsi"][tf] = r
            if r > 50:
                votes += 1

        signal = votes >= BOT_CONFIG["vote_required"]

        if signal and STATE["position"] is None and BOT_CONFIG["enabled"]:
            send_telegram(f"BUY CONFIRM? Price {p}")
            STATE["position"] = "LONG"
            STATE["entry"] = p
            STATE["trades"].append({"side":"BUY","price":p,"time":time.time()})

        if STATE["position"] == "LONG":
            pnl = p - STATE["entry"]
            if pnl > 50 or pnl < -30:
                STATE["pnl"] += pnl
                STATE["wins"] += 1 if pnl>0 else 0
                STATE["losses"] += 1 if pnl<=0 else 0
                STATE["trades"].append({"side":"SELL","price":p,"time":time.time()})
                STATE["position"] = None

        payload = json.dumps({"price":p,"rsi":STATE["rsi"],"state":STATE})
        for ws in clients:
            await ws.send_text(payload)

        await asyncio.sleep(2)

@app.on_event("startup")
async def start_bg():
    asyncio.create_task(price_stream())

# ---------------- API ----------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except:
        clients.remove(ws)

@app.post("/bot/start")
def start():
    BOT_CONFIG["enabled"] = True
    return {"ok":True}

@app.post("/bot/stop")
def stop():
    BOT_CONFIG["enabled"] = False
    return {"ok":True}

@app.get("/bot/status")
def status():
    return STATE

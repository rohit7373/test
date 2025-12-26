import os, time, threading, uuid
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

exchange = ccxt.binance({"enableRateLimit": True})
SYMBOL = "BTC/USDT"

STATE = {
    "running": False,
    "wallet": 1000.0,
    "unrealized": 0.0,
    "openTrades": [],
    "history": []
}

TF_MAP = {
    "30s": "30s",
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w"
}

class ManualOrder(BaseModel):
    side: str
    qty: float
    type: str
    sl: Optional[float] = None
    tp: Optional[float] = None

class CloseTradeReq(BaseModel):
    id: str

def fetch_candles(tf, limit=300):
    tf = TF_MAP.get(tf, "1m")
    ohlcv = exchange.fetch_ohlcv(SYMBOL, tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    df["time"] = (df["time"] / 1000).astype(int)
    df["rsi"] = ta.rsi(df["close"], length=14)
    df.dropna(inplace=True)
    return df

def last_price():
    return float(exchange.fetch_ticker(SYMBOL)["last"])

def bot_loop():
    while True:
        try:
            price = last_price()

            total = 0
            for t in STATE["openTrades"]:
                diff = price - t["entryPrice"]
                if t["side"] == "SHORT":
                    diff = -diff
                t["pnl"] = diff * t["size"]
                total += t["pnl"]
            STATE["unrealized"] = total

            if STATE["running"]:
                stf1, stf2 = "1m", "5m"
                ltf1, ltf2 = "1h", "4h"

                f1 = fetch_candles(stf1)
                f2 = fetch_candles(stf2)
                t1 = fetch_candles(ltf1)
                t2 = fetch_candles(ltf2)

                if len(f1) < 2 or len(f2) < 2:
                    continue

                f1c, f1p = f1["rsi"].iloc[-1], f1["rsi"].iloc[-2]
                f2c, f2p = f2["rsi"].iloc[-1], f2["rsi"].iloc[-2]

                trend_bull = t1["rsi"].iloc[-1] > t2["rsi"].iloc[-1]
                trend_bear = t1["rsi"].iloc[-1] < t2["rsi"].iloc[-1]

                if any(t.get("auto") for t in STATE["openTrades"]):
                    continue

                if trend_bull and f1p <= f2p and f1c > f2c:
                    open_trade("LONG", price, stf1, stf2, ltf1, ltf2)

                if trend_bear and f1p >= f2p and f1c < f2c:
                    open_trade("SHORT", price, stf1, stf2, ltf1, ltf2)

        except Exception as e:
            print("BOT ERROR:", e)

        time.sleep(2)

def open_trade(side, price, stf1, stf2, ltf1, ltf2):
    STATE["openTrades"].append({
        "id": str(uuid.uuid4())[:8],
        "side": side,
        "size": 0.01,
        "entryPrice": price,
        "pnl": 0,
        "auto": True,
        "stf": f"{stf1}>{stf2}",
        "ltf": f"{ltf1}>{ltf2}",
        "time": datetime.now().isoformat()
    })

threading.Thread(target=bot_loop, daemon=True).start()

@app.get("/api/market")
def market(stf1="1m", stf2="5m", ltf1="1h", ltf2="4h"):
    return {
        "price": last_price(),
        "stf1": fetch_candles(stf1).to_dict("records"),
        "stf2": fetch_candles(stf2).to_dict("records"),
        "ltf1": fetch_candles(ltf1).to_dict("records"),
        "ltf2": fetch_candles(ltf2).to_dict("records"),
        "state": STATE,
        "openTrades": STATE["openTrades"],
        "history": STATE["history"]
    }

@app.post("/api/start")
def start():
    STATE["running"] = True
    return {"status":"started"}

@app.post("/api/stop")
def stop():
    STATE["running"] = False
    return {"status":"stopped"}

@app.post("/api/manual/order")
def manual_order(o: ManualOrder):
    STATE["openTrades"].append({
        "id": str(uuid.uuid4())[:8],
        "side": o.side.upper(),
        "size": o.qty,
        "entryPrice": last_price(),
        "pnl": 0,
        "auto": False,
        "stf": "MANUAL",
        "ltf": "MANUAL",
        "time": datetime.now().isoformat()
    })
    return {"status":"ok"}

@app.post("/api/manual/close")
def close_trade(r: CloseTradeReq):
    for t in STATE["openTrades"]:
        if t["id"] == r.id:
            STATE["openTrades"].remove(t)
            STATE["history"].append(t)
            STATE["wallet"] += t["pnl"]
            return {"status":"closed"}
    raise HTTPException(404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

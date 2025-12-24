import os, time, threading
import pandas as pd
import pandas_ta as ta
import ccxt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    "position": None,
    "entry": 0.0,
    "qty": 0.1,
    "unrealized": 0.0,
    "trades": []
}

TF_MAP = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m",
    "1h":"1h","4h":"4h","1d":"1d","1w":"1w"
}

def fetch_candles(tf, limit=300):
    tf = TF_MAP.get(tf, "1m")
    ohlcv = exchange.fetch_ohlcv(SYMBOL, tf, limit=limit)
    df = pd.DataFrame(
        ohlcv, columns=["time","open","high","low","close","volume"]
    )
    df["time"] = (df["time"] / 1000).astype(int)
    df["rsi"] = ta.rsi(df["close"], length=14)
    df.dropna(inplace=True)
    return df

def last_price():
    return float(exchange.fetch_ticker(SYMBOL)["last"])

def bot_loop():
    while True:
        if STATE["running"]:
            try:
                df_f = fetch_candles("1m")
                df_s = fetch_candles("5m")
                rsi_f = df_f["rsi"].iloc[-1]
                rsi_s = df_s["rsi"].iloc[-1]
                price = last_price()

                if STATE["position"]:
                    diff = price - STATE["entry"]
                    if STATE["position"] == "SHORT":
                        diff = -diff
                    STATE["unrealized"] = round(diff * STATE["qty"], 2)

                if rsi_f > rsi_s and STATE["position"] is None:
                    STATE["position"] = "LONG"
                    STATE["entry"] = price
                    STATE["trades"].insert(0,{
                        "time": datetime.now().strftime("%I:%M:%S %p"),
                        "side":"BUY",
                        "price":price,
                        "pnl":0
                    })

                elif rsi_f < rsi_s and STATE["position"] == "LONG":
                    pnl = (price - STATE["entry"]) * STATE["qty"]
                    STATE["wallet"] += pnl
                    STATE["trades"][0]["pnl"] = round(pnl,2)
                    STATE["position"] = None
                    STATE["unrealized"] = 0

            except Exception as e:
                print("BOT ERROR:", e)

        time.sleep(2)

threading.Thread(target=bot_loop, daemon=True).start()

@app.get("/api/market")
def market(stf1:str, stf2:str, ltf1:str, ltf2:str):
    def pack(df):
        return df[["time","open","high","low","close","rsi"]].to_dict("records")

    return {
        "price": last_price(),
        "stf1": pack(fetch_candles(stf1)),
        "stf2": pack(fetch_candles(stf2)),
        "ltf1": pack(fetch_candles(ltf1)),
        "ltf2": pack(fetch_candles(ltf2)),
        "state": STATE
    }

@app.post("/api/start")
def start():
    STATE["running"] = True
    return {"ok":True}

@app.post("/api/stop")
def stop():
    STATE["running"] = False
    return {"ok":True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))

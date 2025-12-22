import os, time, math, threading, datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client

# ================= APP =================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ================= BINANCE =================
client = Client(
    os.getenv("BINANCE_API_KEY"),
    os.getenv("BINANCE_SECRET_KEY"),
    testnet=True
)
client.API_URL = "https://testnet.binance.vision/api"

TF_MAP = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
}

# ================= CONFIG =================
BOT_CONFIG = {
    "symbol": "BTCUSDT",
    "position_size": 10,
    "rsi_sets": [],
    "rsi_required": 2,
    "cooldown_sec": 300,        # ⏱ min gap between trades
    "max_daily_loss": -50,      # 💰 auto stop PnL
    "enabled": False
}

BOT_STATE = {
    "running": False,
    "position": None,
    "entry": None,
    "last_trade_time": 0,
    "daily_pnl": 0.0,
    "rsi_votes": 0,
    "rsi_status": [],
    "auto_stopped": False,
    "auto_stop_reason": ""
}

TRADES = []

# ================= HELPERS =================
def log_trade(side, price, qty, source):
    TRADES.append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "side": side,
        "price": round(price, 2),
        "qty": qty,
        "source": source
    })
    del TRADES[:-100]

def rsi(closes, n):
    g, l = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        g.append(max(d, 0))
        l.append(max(-d, 0))
    ag = sum(g[-n:]) / n
    al = sum(l[-n:]) / n if sum(l[-n:]) else 1e-9
    return 100 - (100 / (1 + ag / al))

def adjust_qty(symbol, qty):
    info = client.get_symbol_info(symbol)
    f = next(x for x in info["filters"] if x["filterType"] == "LOT_SIZE")
    step = float(f["stepSize"])
    minq = float(f["minQty"])
    qty = math.floor(qty / step) * step
    return qty if qty >= minq else None

# ================= RSI COMBO =================
def combo_rsi(symbol, side):
    votes = 0
    status = []

    for idx, s in enumerate(BOT_CONFIG["rsi_sets"]):
        kl = client.get_klines(
            symbol=symbol,
            interval=TF_MAP[s["tf"]],
            limit=s["length"] + 50
        )
        closes = [float(k[4]) for k in kl]
        r = rsi(closes, s["length"])

        passed = False
        if side == "BUY" and r > s["buy"]:
            passed = True
        if side == "SELL" and r < s["sell"]:
            passed = True

        if passed:
            votes += 1

        status.append({
            "set": idx + 1,
            "tf": s["tf"],
            "rsi": round(r, 2),
            "pass": passed
        })

    BOT_STATE["rsi_votes"] = votes
    BOT_STATE["rsi_status"] = status
    return votes >= BOT_CONFIG["rsi_required"]

# ================= BOT LOOP =================
def bot_loop():
    BOT_STATE["running"] = True

    while BOT_CONFIG["enabled"]:
        try:
            now = time.time()
            if now - BOT_STATE["last_trade_time"] < BOT_CONFIG["cooldown_sec"]:
                time.sleep(5)
                continue

            if BOT_STATE["daily_pnl"] <= BOT_CONFIG["max_daily_loss"]:
                BOT_STATE["auto_stopped"] = True
                BOT_STATE["auto_stop_reason"] = "Max daily loss hit"
                BOT_CONFIG["enabled"] = False
                break

            symbol = BOT_CONFIG["symbol"]
            price = float(client.get_symbol_ticker(symbol=symbol)["price"])

            acct = client.get_account()
            bal = {b["asset"]: float(b["free"]) for b in acct["balances"]}
            usdt = bal.get("USDT", 0)
            asset = symbol.replace("USDT", "")

            if BOT_STATE["position"] is None:
                if combo_rsi(symbol, "BUY"):
                    qty = adjust_qty(symbol, (usdt * BOT_CONFIG["position_size"] / 100) / price)
                    if qty:
                        client.create_order(
                            symbol=symbol,
                            side=Client.SIDE_BUY,
                            type=Client.ORDER_TYPE_MARKET,
                            quantity=qty
                        )
                        BOT_STATE["position"] = "BUY"
                        BOT_STATE["entry"] = price
                        BOT_STATE["last_trade_time"] = now
                        log_trade("BUY", price, qty, "BOT")

            elif BOT_STATE["position"] == "BUY":
                if combo_rsi(symbol, "SELL"):
                    qty = adjust_qty(symbol, bal.get(asset, 0))
                    if qty:
                        client.create_order(
                            symbol=symbol,
                            side=Client.SIDE_SELL,
                            type=Client.ORDER_TYPE_MARKET,
                            quantity=qty
                        )
                        pnl = (price - BOT_STATE["entry"]) * qty
                        BOT_STATE["daily_pnl"] += pnl
                        BOT_STATE["position"] = None
                        BOT_STATE["entry"] = None
                        BOT_STATE["last_trade_time"] = now
                        log_trade("SELL", price, qty, "BOT")

        except Exception as e:
            print("BOT ERROR:", e)

        time.sleep(30)

    BOT_STATE["running"] = False

# ================= API =================
@app.post("/bot/config")
async def config(req: Request):
    data = await req.json()
    BOT_CONFIG.update(data)
    return BOT_CONFIG

@app.post("/bot/start")
def start():
    if not BOT_STATE["running"]:
        BOT_CONFIG["enabled"] = True
        BOT_STATE["auto_stopped"] = False
        BOT_STATE["auto_stop_reason"] = ""
        threading.Thread(target=bot_loop, daemon=True).start()
    return {"status": "started"}

@app.post("/bot/stop")
def stop():
    BOT_CONFIG["enabled"] = False
    return {"status": "stopped"}

@app.get("/bot/status")
def status():
    return BOT_STATE | {
        "cooldown_left": max(
            0,
            BOT_CONFIG["cooldown_sec"] - (time.time() - BOT_STATE["last_trade_time"])
        )
    }

@app.get("/bot/trades")
def trades():
    return TRADES

# ================= BACKTEST =================
@app.post("/bot/backtest")
async def backtest(req: Request):
    cfg = await req.json()
    symbol = cfg["symbol"]
    kl = client.get_klines(
        symbol=symbol,
        interval=TF_MAP[cfg["tf"]],
        start_str=cfg["start"],
        end_str=cfg["end"]
    )

    closes = [float(k[4]) for k in kl]
    balance = 1000
    entry = None

    for i in range(len(closes)):
        price = closes[i]
        if entry is None:
            entry = price
        else:
            balance += price - entry
            entry = None

    return {"final_balance": round(balance, 2)}

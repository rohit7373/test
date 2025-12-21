from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client
import os

# ✅ APP MUST BE DEFINED FIRST
app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ BINANCE TESTNET CLIENT
client = Client(
    os.getenv("BINANCE_API_KEY"),
    os.getenv("BINANCE_SECRET_KEY"),
    testnet=True
)
client.API_URL = "https://testnet.binance.vision/api"

# ✅ ROOT CHECK
@app.get("/")
def root():
    return {"status": "Binance Spot Testnet Bot Running"}

# ✅ WEBHOOK (POST ONLY)
@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()

    symbol = data.get("symbol", "BTCUSDT")
    action = data.get("action", "BUY")
    position_pct = float(data.get("position_size", 1))

    account = client.get_account()
    usdt_balance = float(
        next(b for b in account["balances"] if b["asset"] == "USDT")["free"]
    )

    price = float(client.get_symbol_ticker(symbol=symbol)["price"])
    quantity = round((usdt_balance * position_pct / 100) / price, 6)

    side = Client.SIDE_BUY if action == "BUY" else Client.SIDE_SELL

    order = client.create_order(
        symbol=symbol,
        side=side,
        type=Client.ORDER_TYPE_MARKET,
        quantity=quantity
    )

    return {
        "status": "order_sent",
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "order_id": order.get("orderId")
    }

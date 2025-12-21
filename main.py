from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client
from binance.exceptions import BinanceAPIException
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Binance Spot Testnet Bot Running"}

@app.post("/webhook")
async def webhook(req: Request):
    try:
        data = await req.json()

        symbol = data.get("symbol", "BTCUSDT")
        action = data.get("action", "BUY")
        position_pct = float(data.get("position_size", 1))

        # ✅ Create client INSIDE request (prevents startup crash)
        client = Client(
            os.getenv("BINANCE_API_KEY"),
            os.getenv("BINANCE_SECRET_KEY"),
            testnet=True
        )
        client.API_URL = "https://testnet.binance.vision/api"

        account = client.get_account()
        balances = {b["asset"]: float(b["free"]) for b in account["balances"]}

        if "USDT" not in balances or balances["USDT"] <= 0:
            return {
                "status": "error",
                "message": "No USDT balance available in Spot Testnet wallet"
            }

        usdt_balance = balances["USDT"]

        price = float(client.get_symbol_ticker(symbol=symbol)["price"])
        quantity = round((usdt_balance * position_pct / 100) / price, 6)

        if quantity <= 0:
            return {
                "status": "error",
                "message": "Calculated quantity is zero"
            }

        side = Client.SIDE_BUY if action == "BUY" else Client.SIDE_SELL

        order = client.create_order(
            symbol=symbol,
            side=side,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity
        )

        return {
            "status": "success",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_id": order.get("orderId")
        }

    except BinanceAPIException as e:
        return {
            "status": "error",
            "message": f"Binance API error: {e.message}"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Server error: {str(e)}"
        }

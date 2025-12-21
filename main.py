from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client
from binance.exceptions import BinanceAPIException
import os
import math

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Health Check
# ----------------------------
@app.get("/")
def root():
    return {"status": "Binance Spot Testnet Bot Running"}

# ----------------------------
# Helper: Adjust Quantity to LOT_SIZE
# ----------------------------
def adjust_quantity(client: Client, symbol: str, raw_qty: float):
    info = client.get_symbol_info(symbol)
    lot_filter = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")

    step_size = float(lot_filter["stepSize"])
    min_qty = float(lot_filter["minQty"])

    # Floor to step size
    precision = int(round(-math.log(step_size, 10), 0))
    qty = math.floor(raw_qty / step_size) * step_size
    qty = round(qty, precision)

    if qty < min_qty:
        return None, f"Quantity {qty} is below minimum lot size {min_qty}"

    return qty, None

# ----------------------------
# Webhook Endpoint
# ----------------------------
@app.post("/webhook")
async def webhook(req: Request):
    try:
        data = await req.json()

        symbol = data.get("symbol", "BTCUSDT")
        action = data.get("action", "BUY").upper()
        position_pct = float(data.get("position_size", 1))

        # ----------------------------
        # Binance Client (INSIDE request)
        # ----------------------------
        client = Client(
            os.getenv("BINANCE_API_KEY"),
            os.getenv("BINANCE_SECRET_KEY"),
            testnet=True
        )
        client.API_URL = "https://testnet.binance.vision/api"

        # ----------------------------
        # Get USDT Balance
        # ----------------------------
        account = client.get_account()
        balances = {b["asset"]: float(b["free"]) for b in account["balances"]}

        if "USDT" not in balances or balances["USDT"] <= 0:
            return {
                "status": "error",
                "message": "No USDT balance available in Spot Testnet wallet"
            }

        usdt_balance = balances["USDT"]

        # ----------------------------
        # Get Price
        # ----------------------------
        price = float(client.get_symbol_ticker(symbol=symbol)["price"])

        # ----------------------------
        # Calculate Raw Quantity
        # ----------------------------
        raw_qty = (usdt_balance * position_pct / 100) / price

        qty, error = adjust_quantity(client, symbol, raw_qty)
        if error:
            return {
                "status": "error",
                "message": error
            }

        # ----------------------------
        # Place Order
        # ----------------------------
        side = Client.SIDE_BUY if action == "BUY" else Client.SIDE_SELL

        order = client.create_order(
            symbol=symbol,
            side=side,
            type=Client.ORDER_TYPE_MARKET,
            quantity=qty
        )

        return {
            "status": "success",
            "symbol": symbol,
            "action": action,
            "quantity": qty,
            "price": price,
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

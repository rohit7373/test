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
    return {"status": "Binance SPOT Testnet Bot Running"}

# ----------------------------
# Helper: LOT_SIZE handling
# ----------------------------
def adjust_quantity(client: Client, symbol: str, raw_qty: float):
    info = client.get_symbol_info(symbol)
    lot_filter = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")

    step_size = float(lot_filter["stepSize"])
    min_qty = float(lot_filter["minQty"])

    precision = int(round(-math.log(step_size, 10), 0))
    qty = math.floor(raw_qty / step_size) * step_size
    qty = round(qty, precision)

    if qty < min_qty:
        return None, f"Quantity {qty} is below minimum lot size {min_qty}"

    return qty, None

# ----------------------------
# Webhook (SPOT)
# ----------------------------
@app.post("/webhook")
async def webhook(req: Request):
    try:
        data = await req.json()

        action = data.get("action", "BUY").upper()
        symbol = data.get("symbol", "BTCUSDT")
        position_pct = float(data.get("position_size", 10))
        tp_pct = float(data.get("tp_pct", 0))
        sl_pct = float(data.get("sl_pct", 0))

        # ----------------------------
        # Binance Spot Testnet Client
        # ----------------------------
        client = Client(
            os.getenv("BINANCE_API_KEY"),
            os.getenv("BINANCE_SECRET_KEY"),
            testnet=True
        )
        client.API_URL = "https://testnet.binance.vision/api"

        # ----------------------------
        # BUY LOGIC
        # ----------------------------
        if action == "BUY":
            account = client.get_account()
            balances = {b["asset"]: float(b["free"]) for b in account["balances"]}

            if "USDT" not in balances or balances["USDT"] <= 0:
                return {"status": "error", "message": "No USDT balance"}

            usdt_balance = balances["USDT"]
            price = float(client.get_symbol_ticker(symbol=symbol)["price"])

            raw_qty = (usdt_balance * position_pct / 100) / price
            qty, err = adjust_quantity(client, symbol, raw_qty)
            if err:
                return {"status": "error", "message": err}

            buy_order = client.create_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=qty
            )

            # Optional TP / SL
            if tp_pct > 0:
                tp_price = round(price * (1 + tp_pct / 100), 2)
                client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_LIMIT,
                    quantity=qty,
                    price=str(tp_price),
                    timeInForce=Client.TIME_IN_FORCE_GTC
                )

            if sl_pct > 0:
                sl_price = round(price * (1 - sl_pct / 100), 2)
                client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_STOP_LOSS_LIMIT,
                    quantity=qty,
                    stopPrice=str(sl_price),
                    price=str(sl_price),
                    timeInForce=Client.TIME_IN_FORCE_GTC
                )

            return {
                "status": "success",
                "action": "BUY",
                "symbol": symbol,
                "quantity": qty,
                "price": price
            }

        # ----------------------------
        # SELL LOGIC
        # ----------------------------
        elif action == "SELL":
            asset = symbol.replace("USDT", "")
            account = client.get_account()
            balances = {b["asset"]: float(b["free"]) for b in account["balances"]}

            if asset not in balances or balances[asset] <= 0:
                return {"status": "error", "message": f"No {asset} balance"}

            qty = balances[asset] * position_pct / 100

            qty, err = adjust_quantity(client, symbol, qty)
            if err:
                return {"status": "error", "message": err}

            sell_order = client.create_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                type=Client.ORDER_TYPE_MARKET,
                quantity=qty
            )

            return {
                "status": "success",
                "action": "SELL",
                "symbol": symbol,
                "quantity": qty
            }

        else:
            return {"status": "error", "message": "Invalid action"}

    except BinanceAPIException as e:
        return {"status": "error", "message": f"Binance API error: {e.message}"}

    except Exception as e:
        return {"status": "error", "message": f"Server error: {str(e)}"}

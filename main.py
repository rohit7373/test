@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()

    symbol = data.get("symbol", "BTCUSDT")
    action = data.get("action", "BUY")
    position_pct = float(data.get("position_size", 1))

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

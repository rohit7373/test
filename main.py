
import os, time, math, threading
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET_KEY"), testnet=True)
client.API_URL = "https://testnet.binance.vision/api"

TF_MAP={
 "1m":Client.KLINE_INTERVAL_1MINUTE,
 "5m":Client.KLINE_INTERVAL_5MINUTE,
 "15m":Client.KLINE_INTERVAL_15MINUTE,
 "1h":Client.KLINE_INTERVAL_1HOUR,
 "4h":Client.KLINE_INTERVAL_4HOUR,
}

BOT_CONFIG={
 "symbol":"BTCUSDT",
 "timeframe":"5m",
 "rsi_length":14,
 "buy_rsi":70,
 "sell_rsi":40,
 "position_size":10,
 "tp_pct":1,
 "sl_pct":0.5,
 "htf_list":["15m","1h"],
 "votes_required":2,
 "enabled":False
}

BOT_STATE={"running":False,"position":None,"entry":None,"last_rsi":None}

def rsi(closes,n):
 g,l=[],[]
 for i in range(1,len(closes)):
  d=closes[i]-closes[i-1]
  g.append(max(d,0));l.append(max(-d,0))
 ag=sum(g[-n:])/n; al=sum(l[-n:])/n if sum(l[-n:]) else 1e-9
 return 100-(100/(1+ag/al))

def adjust_qty(symbol,qty):
 info=client.get_symbol_info(symbol)
 f=next(x for x in info["filters"] if x["filterType"]=="LOT_SIZE")
 step=float(f["stepSize"]); minq=float(f["minQty"])
 qty=math.floor(qty/step)*step
 return qty if qty>=minq else None

def multi_tf_vote(symbol, length):
 votes=0
 for tf in BOT_CONFIG["htf_list"]:
  kl=client.get_klines(symbol=symbol,interval=TF_MAP[tf],limit=length+50)
  closes=[float(k[4]) for k in kl]
  if rsi(closes,length)>BOT_CONFIG["buy_rsi"]:
   votes+=1
 return votes>=BOT_CONFIG["votes_required"]

def bot_loop():
 BOT_STATE["running"]=True
 while BOT_CONFIG["enabled"]:
  try:
   kl=client.get_klines(symbol=BOT_CONFIG["symbol"],interval=TF_MAP[BOT_CONFIG["timeframe"]],limit=200)
   closes=[float(k[4]) for k in kl]
   price=closes[-1]
   r=rsi(closes,BOT_CONFIG["rsi_length"])
   BOT_STATE["last_rsi"]=round(r,2)

   acct=client.get_account()
   bal={b["asset"]:float(b["free"]) for b in acct["balances"]}
   usdt=bal.get("USDT",0); asset=BOT_CONFIG["symbol"].replace("USDT","")

   if BOT_STATE["position"] is None:
    if r>BOT_CONFIG["buy_rsi"] and multi_tf_vote(BOT_CONFIG["symbol"],BOT_CONFIG["rsi_length"]):
     qty=adjust_qty(BOT_CONFIG["symbol"],(usdt*BOT_CONFIG["position_size"]/100)/price)
     if qty:
      client.create_order(symbol=BOT_CONFIG["symbol"],side=Client.SIDE_BUY,type=Client.ORDER_TYPE_MARKET,quantity=qty)
      BOT_STATE["position"]="BUY"; BOT_STATE["entry"]=price

   elif BOT_STATE["position"]=="BUY":
    if r<BOT_CONFIG["sell_rsi"]:
     qty=adjust_qty(BOT_CONFIG["symbol"],bal.get(asset,0))
     if qty:
      client.create_order(symbol=BOT_CONFIG["symbol"],side=Client.SIDE_SELL,type=Client.ORDER_TYPE_MARKET,quantity=qty)
      BOT_STATE["position"]=None; BOT_STATE["entry"]=None

  except Exception as e:
   print("Bot error",e)

  time.sleep(300)
 BOT_STATE["running"]=False
 
@app.post("/bot/start")
def start():
 if not BOT_STATE["running"]:
  BOT_CONFIG["enabled"]=True
  threading.Thread(target=bot_loop,daemon=True).start()
 return {"status":"started"}

@app.post("/bot/stop")
def stop():
 BOT_CONFIG["enabled"]=False
 return {"status":"stopped"}

@app.post("/bot/config")
async def config(req:Request):
 BOT_CONFIG.update(await req.json())
 return BOT_CONFIG

# In main.py:
@app.get("/bot/status")
def status():
    """Returns the current status of the bot state."""
    return BOT_STATE

# Add this code block to your main.py file, 
# for example, after the existing @app.post("/bot/config") endpoint.



# Add this to your main.py file

@app.post("/trade/buy")
def manual_buy():
    try:
        ticker = client.get_symbol_ticker(symbol=BOT_CONFIG["symbol"])
        price = float(ticker["price"])

        acct = client.get_account()
        usdt = next(float(b["free"]) for b in acct["balances"] if b["asset"] == "USDT")

        qty = adjust_qty(
            BOT_CONFIG["symbol"],
            (usdt * BOT_CONFIG["position_size"] / 100) / price
        )

        if not qty:
            return {"status": "Error", "message": "Insufficient balance"}

        client.create_order(
            symbol=BOT_CONFIG["symbol"],
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=qty
        )

        BOT_STATE["position"] = "BUY"
        BOT_STATE["entry"] = price

        return {"status": "BUY placed", "price": price, "qty": qty}

    except Exception as e:
        return {"status": "Error", "message": str(e)}


@app.post("/trade/sell")
def manual_sell():
    try:
        asset = BOT_CONFIG["symbol"].replace("USDT", "")
        acct = client.get_account()
        qty = next(float(b["free"]) for b in acct["balances"] if b["asset"] == asset)

        qty = adjust_qty(BOT_CONFIG["symbol"], qty)
        if not qty:
            return {"status": "Error", "message": "No asset to sell"}

        client.create_order(
            symbol=BOT_CONFIG["symbol"],
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=qty
        )

        BOT_STATE["position"] = None
        BOT_STATE["entry"] = None

        return {"status": "SELL placed", "qty": qty}

    except Exception as e:
        return {"status": "Error", "message": str(e)}


import os, math
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client
from datetime import datetime

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET_KEY"), testnet=True)
client.API_URL = "https://testnet.binance.vision/api"

TF_MAP = {
 "1m":"1m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","2h":"2h","4h":"4h"
}

def rsi(closes, n=14):
 gains,losses=[],[]
 for i in range(1,len(closes)):
  d=closes[i]-closes[i-1]
  gains.append(max(d,0));losses.append(max(-d,0))
 ag=sum(gains[-n:])/n; al=sum(losses[-n:])/n if sum(losses[-n:]) else 1e-9
 return 100-(100/(1+ag/al))

def ema(vals,p):
 k=2/(p+1); e=vals[0]
 for v in vals[1:]: e=v*k+e*(1-k)
 return e

def backtest(symbol, tf, start, end, capital, params):
 kl=client.get_historical_klines(symbol, tf, start, end)
 closes=[float(k[4]) for k in kl]
 bal=capital; pos=None; entry=0; peak=capital
 trades=0; wins=0

 for i in range(60,len(closes)):
  price=closes[i]
  r=rsi(closes[:i],params["rsi_len"])
  e=ema(closes[i-60:i],50)

  if pos is None and r>params["buy"] and price>e:
   pos=price; entry=price

  elif pos:
   tp=entry*(1+params["tp"]/100); sl=entry*(1-params["sl"]/100)
   if r<params["sell"] or price>=tp or price<=sl:
    pnl=(price-entry)/entry
    bal*=1+pnl; trades+=1; wins+=pnl>0
    pos=None; peak=max(peak,bal)

 dd=(peak-bal)/peak*100
 score=(bal-capital)/capital*100 - dd + (wins/max(1,trades))*10

 return {"end_balance":round(bal,2),"trades":trades,"win_rate":round(wins/max(1,trades)*100,2),
         "drawdown":round(dd,2),"ai_score":round(score,2),"params":params}

@app.post("/backtest/run")
async def run(req:Request):
 d=await req.json()
 p={"rsi_len":14,"buy":70,"sell":40,"tp":1,"sl":0.5}
 return backtest(d["symbol"],TF_MAP[d["timeframe"]],d["start"],d["end"],d["capital"],p)

@app.post("/backtest/optimize")
async def optimize(req:Request):
 d=await req.json()
 best=None
 for rsi_len in [10,14,21]:
  for buy in [65,70,75]:
   for sell in [35,40,45]:
    p={"rsi_len":rsi_len,"buy":buy,"sell":sell,"tp":1,"sl":0.5}
    res=backtest(d["symbol"],TF_MAP[d["timeframe"]],d["start"],d["end"],d["capital"],p)
    if not best or res["ai_score"]>best["ai_score"]:
     best=res
 return best

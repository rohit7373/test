import threading
import time
import pandas as pd
import pandas_ta as ta
import ccxt
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import os

app = Flask(__name__)

# --- ENABLE CORS ---
# This allows your Netlify frontend to talk to this Railway backend
CORS(app) 

# --- CONFIGURATION & STATE ---
BOT_STATE = {
    "is_running": False,
    "wallet_balance": 1000.00,
    "positions": {"bot": None, "manual": None}, 
    "trades": [], 
}

BOT_CONFIG = {
    "symbol": "BTC/USDT",
    "strategy_mode": "BOTH", 
    "stf1": "1m", "stf2": "5m", "stf_logic": ">",
    "ltf1": "4h", "ltf2": "1h", "ltf_logic": ">",
    "bot_qty": 0.1, "bot_tp": 1.5, "bot_sl": 1.0
}

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# --- HELPER FUNCTIONS ---

def fetch_rsi_data(symbol, timeframe, limit=50):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv: return []
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['rsi'] = ta.rsi(df['close'], length=14)
        df = df.dropna()
        
        data = []
        for index, row in df.iterrows():
            data.append({
                'time': int(row['timestamp'].timestamp()),
                'open': row['open'], 'high': row['high'],
                'low': row['low'], 'close': row['close'],
                'volume': row['volume'], 'rsi': row['rsi']
            })
        return data
    except Exception as e:
        print(f"Error fetching {timeframe}: {e}")
        return []

def execute_trade(source, side, qty, price):
    if BOT_STATE["positions"][source] is not None:
        close_position(source, price)

    pos_type = "LONG" if side == "BUY" else "SHORT"
    BOT_STATE["positions"][source] = {
        "type": pos_type, "entry": price, "qty": qty
    }
    log_trade(source, f"{side} ({pos_type})", qty, price)
    print(f"[{source.upper()}] Executed {side} at {price}")

def close_position(source, price):
    pos = BOT_STATE["positions"][source]
    if not pos: return

    diff = price - pos["entry"]
    if pos["type"] == "SHORT": diff = -diff
    pnl = diff * pos["qty"]
    
    BOT_STATE["wallet_balance"] += pnl
    log_trade(source, "CLOSE", pos["qty"], price, pnl)
    BOT_STATE["positions"][source] = None

def log_trade(source, side, qty, price, pnl=None):
    rec = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source, "side": side, "qty": qty,
        "price": price, "pnl": pnl
    }
    BOT_STATE["trades"].insert(0, rec)

# --- BOT LOGIC LOOP ---

def bot_loop():
    print("Background Bot Thread Started...")
    while True:
        if BOT_STATE["is_running"]:
            try:
                s1 = fetch_rsi_data(BOT_CONFIG["symbol"], BOT_CONFIG["stf1"], 20)
                s2 = fetch_rsi_data(BOT_CONFIG["symbol"], BOT_CONFIG["stf2"], 20)
                l1 = fetch_rsi_data(BOT_CONFIG["symbol"], BOT_CONFIG["ltf1"], 20)
                l2 = fetch_rsi_data(BOT_CONFIG["symbol"], BOT_CONFIG["ltf2"], 20)

                if s1 and s2 and l1 and l2:
                    price = s1[-1]['close']
                    r_s1 = s1[-1]['rsi']
                    r_s2 = s2[-1]['rsi']
                    r_l1 = l1[-1]['rsi']
                    r_l2 = l2[-1]['rsi']

                    stf_ok = (r_s1 > r_s2) if BOT_CONFIG["stf_logic"] == ">" else (r_s1 < r_s2)
                    ltf_ok = (r_l1 > r_l2) if BOT_CONFIG["ltf_logic"] == ">" else (r_l1 < r_l2)

                    trigger = False
                    mode = BOT_CONFIG["strategy_mode"]
                    if mode == "1" and stf_ok: trigger = True
                    elif mode == "2" and ltf_ok: trigger = True
                    elif mode == "BOTH" and stf_ok and ltf_ok: trigger = True

                    if trigger and BOT_STATE["positions"]["bot"] is None:
                        execute_trade("bot", "BUY", BOT_CONFIG["bot_qty"], price)
            except Exception as e:
                print(f"Bot Loop Error: {e}")
        time.sleep(2)

t = threading.Thread(target=bot_loop, daemon=True)
t.start()

# --- API ENDPOINTS ---

@app.route('/')
def index():
    # Since HTML is on Netlify, the root URL just confirms the server is up.
    return jsonify({"status": "Bot Backend is Running", "frontend": "Hosted on Netlify"})

@app.route('/api/market_data', methods=['GET'])
def get_data():
    symbol = request.args.get('symbol', 'BTC/USDT')
    
    if not BOT_STATE["is_running"]:
        BOT_CONFIG["stf1"] = request.args.get('stf1', '1m')
        BOT_CONFIG["stf2"] = request.args.get('stf2', '5m')
        BOT_CONFIG["ltf1"] = request.args.get('ltf1', '4h')
        BOT_CONFIG["ltf2"] = request.args.get('ltf2', '1h')
        BOT_CONFIG["stf_logic"] = request.args.get('stf_logic', '>')
        BOT_CONFIG["ltf_logic"] = request.args.get('ltf_logic', '>')

    stf1 = fetch_rsi_data(symbol, BOT_CONFIG["stf1"])
    stf2 = fetch_rsi_data(symbol, BOT_CONFIG["stf2"])
    ltf1 = fetch_rsi_data(symbol, BOT_CONFIG["ltf1"])
    ltf2 = fetch_rsi_data(symbol, BOT_CONFIG["ltf2"])
    
    curr_price = stf1[-1]['close'] if stf1 else 0

    def calc_pnl(pos):
        if not pos: return 0.0
        diff = curr_price - pos['entry']
        if pos['type'] == 'SHORT': diff = -diff
        return diff * pos['qty']

    return jsonify({
        'stf1': stf1, 'stf2': stf2,
        'ltf1': ltf1, 'ltf2': ltf2,
        'current_price': curr_price,
        'state': {
            'is_running': BOT_STATE["is_running"],
            'wallet': BOT_STATE["wallet_balance"],
            'bot_pnl': calc_pnl(BOT_STATE["positions"]["bot"]),
            'manual_pnl': calc_pnl(BOT_STATE["positions"]["manual"]),
            'trades': BOT_STATE["trades"]
        }
    })

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    data = request.json
    BOT_CONFIG["strategy_mode"] = data.get("strategy_mode", "BOTH")
    BOT_CONFIG["bot_qty"] = float(data.get("bot_qty", 0.1))
    BOT_STATE["is_running"] = True
    return jsonify({"status": "started"})

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    BOT_STATE["is_running"] = False
    return jsonify({"status": "stopped"})

@app.route('/api/manual/trade', methods=['POST'])
def manual_trade():
    d = request.json
    execute_trade("manual", d['side'], float(d['qty']), float(d['price']))
    return jsonify({"status": "executed"})

@app.route('/api/close_all', methods=['POST'])
def close_all():
    price = float(request.json.get('price', 0))
    close_position("bot", price)
    close_position("manual", price)
    return jsonify({"status": "closed"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

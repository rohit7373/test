import threading
import time
import pandas as pd
import pandas_ta as ta
import ccxt
from flask import Flask, render_template, jsonify, request
from datetime import datetime

app = Flask(__name__)

# --- CONFIGURATION & STATE ---
# We use a global dictionary to store the state of the bot and wallet
BOT_STATE = {
    "is_running": False,
    "wallet_balance": 1000.00,
    "positions": {"bot": None, "manual": None}, # Stores { 'type': 'LONG', 'entry': 50000, 'qty': 0.1 }
    "trades": [], # History
    "last_check_time": 0
}

# Default Configuration (Updated by Frontend)
BOT_CONFIG = {
    "symbol": "BTC/USDT",
    "strategy_mode": "BOTH", # 1, 2, or BOTH
    "stf1": "1m", "stf2": "5m", "stf_logic": ">",
    "ltf1": "4h", "ltf2": "1h", "ltf_logic": ">",
    "bot_qty": 0.1, "bot_tp": 1.5, "bot_sl": 1.0
}

# Initialize Exchange (CCXT for Data)
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# --- HELPER FUNCTIONS ---

def fetch_rsi_data(symbol, timeframe, limit=100):
    """Fetches OHLCV and calculates RSI."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv: return []
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        df = df.dropna()
        
        # Convert to list of dicts for lightweight-charts
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
    """Updates the internal wallet and position state."""
    # source: 'bot' or 'manual'
    # side: 'BUY' (Long) or 'SELL' (Short) -- simplified for this logic
    
    # 1. Close existing position if any (Simple Reversal Logic)
    if BOT_STATE["positions"][source] is not None:
        close_position(source, price)

    # 2. Open new position
    pos_type = "LONG" if side == "BUY" else "SHORT"
    BOT_STATE["positions"][source] = {
        "type": pos_type,
        "entry": price,
        "qty": qty
    }
    
    # 3. Log Trade
    trade_record = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "side": f"{side} ({pos_type})",
        "qty": qty,
        "price": price,
        "source": source
    }
    BOT_STATE["trades"].insert(0, trade_record) # Add to top
    print(f"[{source.upper()}] Executed {side} at {price}")

def close_position(source, current_price):
    """Calculates PnL and updates wallet."""
    pos = BOT_STATE["positions"][source]
    if not pos: return

    # Calculate PnL
    diff = current_price - pos["entry"]
    if pos["type"] == "SHORT": diff = -diff
    
    pnl = diff * pos["qty"]
    BOT_STATE["wallet_balance"] += pnl
    
    # Log Closing
    trade_record = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "side": "CLOSE",
        "qty": pos["qty"],
        "price": current_price,
        "source": source,
        "pnl": round(pnl, 2)
    }
    BOT_STATE["trades"].insert(0, trade_record)
    BOT_STATE["positions"][source] = None # Reset
    print(f"[{source.upper()}] Closed Position. PnL: {pnl:.2f}")

# --- BACKGROUND BOT LOOP ---

def bot_logic_loop():
    """Runs in a separate thread to check conditions continuously."""
    print("Background Bot Thread Started")
    while True:
        if BOT_STATE["is_running"]:
            try:
                # 1. Fetch Latest Data (Just the last few candles needed for logic)
                # We use the TFs stored in BOT_CONFIG
                s1 = fetch_rsi_data(BOT_CONFIG["symbol"], BOT_CONFIG["stf1"], 20)
                s2 = fetch_rsi_data(BOT_CONFIG["symbol"], BOT_CONFIG["stf2"], 20)
                l1 = fetch_rsi_data(BOT_CONFIG["symbol"], BOT_CONFIG["ltf1"], 20)
                l2 = fetch_rsi_data(BOT_CONFIG["symbol"], BOT_CONFIG["ltf2"], 20)

                if s1 and s2 and l1 and l2:
                    current_price = s1[-1]['close']
                    
                    # 2. Get RSI Values
                    rsi_stf1 = s1[-1]['rsi']
                    rsi_stf2 = s2[-1]['rsi']
                    rsi_ltf1 = l1[-1]['rsi']
                    rsi_ltf2 = l2[-1]['rsi']

                    # 3. Check Logic
                    # STF Logic
                    stf_true = (rsi_stf1 > rsi_stf2) if BOT_CONFIG["stf_logic"] == ">" else (rsi_stf1 < rsi_stf2)
                    # LTF Logic
                    ltf_true = (rsi_ltf1 > rsi_ltf2) if BOT_CONFIG["ltf_logic"] == ">" else (rsi_ltf1 < rsi_ltf2)

                    # 4. Strategy Decision
                    trigger = False
                    mode = BOT_CONFIG["strategy_mode"]
                    
                    if mode == "1" and stf_true: trigger = True
                    elif mode == "2" and ltf_true: trigger = True
                    elif mode == "BOTH" and stf_true and ltf_true: trigger = True

                    # 5. Execute Bot Trade if Triggered and No Position
                    # Note: logic dictates direction. If logic is '>' we assume BUY, if '<' SELL.
                    # For V7 demo, we assume the user sets logic > for Buys.
                    if trigger and BOT_STATE["positions"]["bot"] is None:
                        side = "BUY" # Simplification: Assuming crossover means Buy
                        execute_trade("bot", side, BOT_CONFIG["bot_qty"], current_price)
            
            except Exception as e:
                print(f"Bot Loop Error: {e}")
        
        time.sleep(2) # Wait 2 seconds before next check

# Start Thread
t = threading.Thread(target=bot_logic_loop, daemon=True)
t.start()

# --- FLASK ENDPOINTS ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/market_data', methods=['GET'])
def get_market_data():
    """
    1. Fetches data for charts.
    2. Updates Global Config if Bot is NOT running (allows UI to change TFs).
    3. Returns Data + Current State + PnL.
    """
    symbol = request.args.get('symbol', 'BTC/USDT')
    
    # Update Config from UI Params only if bot is stopped (to prevent changing mid-trade)
    if not BOT_STATE["is_running"]:
        BOT_CONFIG["stf1"] = request.args.get('stf1', '1m')
        BOT_CONFIG["stf2"] = request.args.get('stf2', '5m')
        BOT_CONFIG["ltf1"] = request.args.get('ltf1', '4h')
        BOT_CONFIG["ltf2"] = request.args.get('ltf2', '1h')
        BOT_CONFIG["stf_logic"] = request.args.get('stf_logic', '>')
        BOT_CONFIG["ltf_logic"] = request.args.get('ltf_logic', '>')
    
    # Fetch Data for Charts
    stf1_data = fetch_rsi_data(symbol, BOT_CONFIG["stf1"])
    stf2_data = fetch_rsi_data(symbol, BOT_CONFIG["stf2"])
    ltf1_data = fetch_rsi_data(symbol, BOT_CONFIG["ltf1"])
    ltf2_data = fetch_rsi_data(symbol, BOT_CONFIG["ltf2"])
    
    curr_price = stf1_data[-1]['close'] if stf1_data else 0

    # Calculate Unrealized PnL for frontend display
    bot_pnl = 0
    if BOT_STATE["positions"]["bot"]:
        p = BOT_STATE["positions"]["bot"]
        diff = curr_price - p["entry"]
        if p["type"] == "SHORT": diff = -diff
        bot_pnl = diff * p["qty"]

    manual_pnl = 0
    if BOT_STATE["positions"]["manual"]:
        p = BOT_STATE["positions"]["manual"]
        diff = curr_price - p["entry"]
        if p["type"] == "SHORT": diff = -diff
        manual_pnl = diff * p["qty"]

    return jsonify({
        'stf1': stf1_data, 'stf2': stf2_data,
        'ltf1': ltf1_data, 'ltf2': ltf2_data,
        'current_price': curr_price,
        'state': {
            'is_running': BOT_STATE["is_running"],
            'wallet': BOT_STATE["wallet_balance"],
            'bot_pnl': bot_pnl,
            'manual_pnl': manual_pnl,
            'trades': BOT_STATE["trades"]
        }
    })

# --- CONTROL ENDPOINTS (Called by Buttons) ---

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    data = request.json
    # Apply specific settings from the Start Button if needed
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
    data = request.json
    side = data.get("side") # 'BUY' or 'SELL'
    qty = float(data.get("qty", 0.5))
    price = float(data.get("price", 0)) # Frontend should send current price
    
    execute_trade("manual", side, qty, price)
    return jsonify({"status": "executed"})

@app.route('/api/close_all', methods=['POST'])
def close_all():
    data = request.json
    price = float(data.get("price", 0))
    
    close_position("bot", price)
    close_position("manual", price)
    return jsonify({"status": "closed"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)


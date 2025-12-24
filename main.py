<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RSI Bot V7 – Final</title>

<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
<script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>

<style>
:root{
  --bg:#0d1117;--panel:#161b22;--border:#30363d;
  --green:#238636;--red:#da3633;--blue:#58a6ff
}
body{
  margin:0;height:100vh;display:flex;
  background:var(--bg);color:#c9d1d9;
  font-family:Segoe UI,system-ui
}
.sidebar{
  width:340px;background:var(--panel);
  border-right:1px solid var(--border);
  padding:10px;overflow-y:auto
}
.section{
  border:1px solid var(--border);
  border-radius:6px;padding:10px;margin-bottom:10px
}
.title{color:var(--blue);font-weight:bold;margin-bottom:6px}
.row{display:flex;justify-content:space-between;margin-bottom:6px}
select,input,button{
  background:#0d1117;border:1px solid var(--border);
  color:white;padding:6px;border-radius:4px;width:100%
}
button{cursor:pointer;font-weight:bold}
.green{background:var(--green)}
.red{background:var(--red)}
.orange{background:#d29922;color:black}

.main{
  flex:1;display:flex;flex-direction:column;padding:6px;gap:6px
}
.grid4{
  flex:7;display:grid;
  grid-template-columns:1fr 1fr;
  grid-template-rows:1fr 1fr;gap:6px
}
.grid2{
  flex:3;display:grid;
  grid-template-columns:1fr 1fr;gap:6px
}
.chart-box{
  border:1px solid var(--border);
  position:relative
}
.chart-label{
  position:absolute;top:4px;left:6px;
  background:#000a;padding:2px 6px;
  font-size:11px;color:var(--blue);z-index:5
}
.chart{width:100%;height:100%}
</style>
</head>

<body>

<!-- ================= LEFT SIDEBAR (UNCHANGED) ================= -->
<div class="sidebar">

  <div class="section">
    <div class="row"><b>Wallet:</b> $<span id="wallet">1000.00</span></div>
    <div class="row"><span>Bot Unrl PnL:</span><span id="botPnl">0.00</span></div>
    <div class="row"><span>Time:</span><span id="time">--</span></div>
    <div class="row"><span>Price:</span><span id="price">--</span></div>
  </div>

  <div class="section">
    <div class="title">BOT CONFIGURATION (V7)</div>
    <div class="row"><span>Status:</span><span id="status" style="color:red">STOPPED</span></div>

    <div class="row">
      <span>Strategy Mode</span>
      <select id="mode">
        <option>Both (1 & 2)</option>
      </select>
    </div>

    <div class="title">1. STF CONFIRMATION</div>
    <div class="row"><span>TF 1</span><select id="stf1"><option>1m</option><option>5m</option></select></div>
    <div class="row"><span>TF 2</span><select id="stf2"><option>5m</option><option>15m</option></select></div>

    <div class="title">2. LTF CONFIRMATION</div>
    <div class="row"><span>TF 1</span><select id="ltf1"><option>1h</option><option>4h</option></select></div>
    <div class="row"><span>TF 2</span><select id="ltf2"><option>1h</option><option>4h</option></select></div>

    <div class="row"><span>Bot Qty</span><input id="qty" value="0.1"></div>
    <button class="green" onclick="startBot()">START BOT</button>
  </div>

  <div class="section">
    <div class="title">MANUAL OVERRIDE</div>
    <button class="green">BUY / LONG</button>
    <button class="red">SELL / SHORT</button>
    <button class="orange">CLOSE ALL</button>
  </div>

</div>

<!-- ================= MAIN CHART AREA ================= -->
<div class="main">

  <!-- 4 PRICE CHARTS -->
  <div class="grid4">
    <div class="chart-box"><div class="chart-label">STF TF-1</div><div id="c1" class="chart"></div></div>
    <div class="chart-box"><div class="chart-label">STF TF-2</div><div id="c2" class="chart"></div></div>
    <div class="chart-box"><div class="chart-label">LTF TF-1</div><div id="c3" class="chart"></div></div>
    <div class="chart-box"><div class="chart-label">LTF TF-2</div><div id="c4" class="chart"></div></div>
  </div>

  <!-- RSI CHARTS -->
  <div class="grid2">
    <div class="chart-box"><div class="chart-label">STF RSI (TF1 vs TF2)</div><div id="rsi1" class="chart"></div></div>
    <div class="chart-box"><div class="chart-label">LTF RSI (TF1 vs TF2)</div><div id="rsi2" class="chart"></div></div>
  </div>

</div>

<script>
axios.defaults.baseURL="http://localhost:8000";

/* ---------- CHART CREATION ---------- */
function mkChart(id){
  return LightweightCharts.createChart(
    document.getElementById(id),
    {layout:{backgroundColor:'#000',textColor:'#ccc'},timeScale:{timeVisible:true}}
  )
}
const charts={
  c1:mkChart('c1'),c2:mkChart('c2'),c3:mkChart('c3'),c4:mkChart('c4')
}
const candles={
  c1:charts.c1.addCandlestickSeries(),
  c2:charts.c2.addCandlestickSeries(),
  c3:charts.c3.addCandlestickSeries(),
  c4:charts.c4.addCandlestickSeries()
}

const rsi1=mkChart('rsi1'),rsi2=mkChart('rsi2');
const rsi1a=rsi1.addLineSeries({color:'#22d3ee'});
const rsi1b=rsi1.addLineSeries({color:'#f59e0b'});
const rsi2a=rsi2.addLineSeries({color:'#a855f7'});
const rsi2b=rsi2.addLineSeries({color:'#f59e0b'});

/* ---------- RSI CROSS LOGIC ---------- */
function rsiCross(a,b){
  let m=[];
  for(let i=1;i<Math.min(a.length,b.length);i++){
    let p=a[i-1].value-b[i-1].value;
    let c=a[i].value-b[i].value;
    if(p<0&&c>0) m.push({time:a[i].time,shape:'diamond',color:'#22c55e',position:'inBar'});
    if(p>0&&c<0) m.push({time:a[i].time,shape:'diamond',color:'#ef4444',position:'inBar'});
  }
  return m;
}

/* ---------- UPDATE LOOP ---------- */
async function update(){
  const res=await axios.get('/api/market',{
    params:{
      stf1:stf1.value,stf2:stf2.value,
      ltf1:ltf1.value,ltf2:ltf2.value
    }
  });
  const d=res.data;

  document.getElementById('wallet').innerText=d.state.wallet.toFixed(2);
  document.getElementById('botPnl').innerText=d.state.unrealized.toFixed(2);
  document.getElementById('price').innerText=d.price.toFixed(2);
  document.getElementById('time').innerText=new Date().toLocaleTimeString('en-US',{hour12:true});

  candles.c1.setData(d.stf1.map(x=>x));
  candles.c2.setData(d.stf2.map(x=>x));
  candles.c3.setData(d.ltf1.map(x=>x));
  candles.c4.setData(d.ltf2.map(x=>x));

  const s1=d.stf1.map(x=>({time:x.time,value:x.rsi}));
  const s2=d.stf2.map(x=>({time:x.time,value:x.rsi}));
  rsi1a.setData(s1); rsi1b.setData(s2);
  rsi1a.setMarkers(rsiCross(s1,s2));

  const l1=d.ltf1.map(x=>({time:x.time,value:x.rsi}));
  const l2=d.ltf2.map(x=>({time:x.time,value:x.rsi}));
  rsi2a.setData(l1); rsi2b.setData(l2);
  rsi2a.setMarkers(rsiCross(l1,l2));
}

setInterval(update,2000);
update();

function startBot(){
  axios.post('/api/start');
  status.innerText='RUNNING';
  status.style.color='lime';
}
</script>

</body>
</html>

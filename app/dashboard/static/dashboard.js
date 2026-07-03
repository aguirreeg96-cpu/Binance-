/* Paper Trading Dashboard — vanilla JS, no external deps, no code injection.
 * All user data is inserted via textContent or explicit DOM construction,
 * never via innerHTML with API-sourced content.
 */
'use strict';

const API = '/api/v1/paper-breakout';
const REFRESH_MS = 30_000;
const EVENTS_POLL_MS = 15_000;

let _data = null;         // last successfully fetched dashboard summary
let _fetching = false;
let _refreshTimer = null;
let _allSignals = [];     // full signals list for client-side filter
let _signalFilter = '';
let _lastSeenEventId = 0;
let _eventsTimer = null;

/* ------------------------------------------------------------------ init */

document.addEventListener('DOMContentLoaded', () => {
  fetchDashboard();
  fetchHealth();
  fetchEvents();
  _refreshTimer = setInterval(fetchDashboard, REFRESH_MS);
  _eventsTimer = setInterval(() => { fetchHealth(); fetchEvents(); }, EVENTS_POLL_MS);
});

function refreshDashboard() {
  clearInterval(_refreshTimer);
  fetchDashboard();
  fetchHealth();
  fetchEvents();
  _refreshTimer = setInterval(fetchDashboard, REFRESH_MS);
}

/* ------------------------------------------------------------ data fetch */

async function fetchDashboard() {
  if (_fetching) return;
  _fetching = true;
  setUpdating(true);

  try {
    const res = await fetch(API + '/dashboard-summary');
    if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + res.statusText);
    const data = await res.json();
    _data = data;
    clearFetchError();
    render(data);
  } catch (err) {
    showFetchError(String(err));
    // Last data stays visible
  } finally {
    _fetching = false;
    setUpdating(false);
    updateLastRefreshed();
  }
}

/* ------------------------------------------------------------ rendering */

function render(d) {
  renderHeader(d);
  renderWarnings(d.warnings || []);
  renderSummaryCards(d);
  renderCurrentEvaluation(d);
  renderOpenPosition(d);
  _allSignals = d.signals || [];
  renderSignalTable(_filterSignals(_allSignals, _signalFilter));
  renderTradeTable(d.recent_trades || []);
  renderEquityChart(d.equity_curve || [], d.launch ? d.launch.initial_capital : null);
  renderFrozenConfig(d.frozen_config || {}, d.launch);
  renderSystemStatus(d);
}

/* header */
function renderHeader(d) {
  const badge = document.getElementById('launch-status-badge');
  const st = d.launch ? d.launch.status : 'UNKNOWN';
  badge.textContent = st;
  badge.className = 'badge-status ' + ({
    ACTIVE: 'badge-active', STOPPED: 'badge-stopped', ERROR: 'badge-error'
  }[st] || 'badge-unknown');
}

/* warnings */
function renderWarnings(warnings) {
  const bar = document.getElementById('warnings-bar');
  bar.textContent = '';
  if (!warnings || warnings.length === 0) { bar.style.display = 'none'; return; }
  bar.style.display = '';
  warnings.forEach(w => {
    const p = document.createElement('p');
    p.textContent = w;
    bar.appendChild(p);
  });
}

/* summary cards */
function renderSummaryCards(d) {
  const acc = d.account || {};
  const pos = d.position;
  const stats = d.trade_stats || {};
  const signals = d.signals || [];
  const latestSig = signals.length ? signals[0] : null;

  // current signal
  const sigEl = document.getElementById('card-signal');
  const sigVal = latestSig ? latestSig.signal : (d.launch_exists ? 'WAIT' : '—');
  sigEl.textContent = sigVal;
  sigEl.className = 'card-value ' + signalColor(sigVal);

  // balance
  setText('card-balance', acc.balance ? fmt2(acc.balance) : '—');

  // equity
  setText('card-equity', acc.equity ? fmt2(acc.equity) : '—');

  // pnl
  const pnlEl = document.getElementById('card-pnl');
  const pnlVal = acc.realized_pnl || '0';
  pnlEl.textContent = fmt2(pnlVal);
  pnlEl.className = 'card-value ' + pnlColor(pnlVal);
  if (stats.total_realized_pnl_pct !== null && stats.total_realized_pnl_pct !== undefined) {
    setText('card-pnl-pct', stats.total_realized_pnl_pct + '%');
  } else {
    setText('card-pnl-pct', '—');
  }

  // position
  const posEl = document.getElementById('card-position');
  if (pos) {
    posEl.textContent = 'LONG';
    posEl.className = 'card-value green';
  } else {
    posEl.textContent = '—';
    posEl.className = 'card-value';
  }

  // trades
  setText('card-trades-count', stats.total_trades !== undefined ? stats.total_trades : '0');

  // winrate
  setText('card-winrate', stats.win_rate_pct !== null && stats.win_rate_pct !== undefined
    ? stats.win_rate_pct + '%' : '—');
}

/* current evaluation */
function renderCurrentEvaluation(d) {
  const container = document.getElementById('current-eval');
  container.textContent = '';
  const signals = d.signals || [];
  const ev = signals.length ? signals[0] : null; // newest first

  const items = [
    { label: 'Última vela evaluada (UTC)', value: ev ? ev.candle_close_time : '—' },
    { label: 'Hora local', value: ev ? localTime(ev.candle_close_time) : '—' },
    { label: 'Señal', value: ev ? ev.signal : (d.launch_exists ? 'Esperando...' : '—') },
    { label: 'Razones', value: ev ? (ev.reasons || []).join(' · ') || '—' : '—' },
    { label: 'Precio evaluado', value: ev && ev.raw_market_price ? fmt8(ev.raw_market_price) : '—' },
    { label: 'Donchian entrada', value: ev && ev.entry_donchian_level ? fmt2(ev.entry_donchian_level) : '—' },
    { label: 'Donchian salida', value: ev && ev.exit_donchian_level ? fmt2(ev.exit_donchian_level) : '—' },
    { label: 'EMA 200', value: ev && ev.ema_200 ? fmt2(ev.ema_200) : '—' },
    { label: 'Pendiente EMA', value: ev && ev.ema_slope ? fmtN(ev.ema_slope, 4) : '—' },
    { label: 'ATR', value: ev && ev.atr ? fmtN(ev.atr, 2) : '—' },
    { label: 'Próxima evaluación (UTC)', value: d.next_eligible_close_utc ? d.next_eligible_close_utc : '—' },
    { label: 'Próxima evaluación (local)', value: d.next_eligible_close_utc ? localTime(d.next_eligible_close_utc) : '—' },
    { label: 'Hash config (abrev.)', value: ev ? ev.frozen_config_hash.slice(0, 12) + '…' : '—' },
  ];

  items.forEach(({ label, value }) => {
    const div = document.createElement('div');
    div.className = 'eval-item';
    const lbl = document.createElement('div');
    lbl.className = 'eval-label';
    lbl.textContent = label;
    const val = document.createElement('div');
    val.className = 'eval-value';
    val.textContent = value;
    div.appendChild(lbl);
    div.appendChild(val);
    container.appendChild(div);
  });

  // Friendly explanation for WAIT
  if (ev && ev.signal === 'WAIT') {
    const note = document.createElement('div');
    note.className = 'eval-item';
    note.style.gridColumn = '1 / -1';
    const lbl = document.createElement('div');
    lbl.className = 'eval-label';
    lbl.textContent = 'Interpretación';
    const val = document.createElement('div');
    val.className = 'eval-note';
    val.textContent = 'La estrategia analizó la última vela cerrada y no encontró una entrada válida.';
    note.appendChild(lbl);
    note.appendChild(val);
    container.appendChild(note);
  }
}

/* open position */
function renderOpenPosition(d) {
  const container = document.getElementById('position-container');
  container.textContent = '';

  if (!d.position) {
    const p = document.createElement('p');
    p.className = 'empty-state';
    p.textContent = 'No hay una posición virtual abierta.';
    container.appendChild(p);
    return;
  }

  const pos = d.position;
  const fields = [
    { label: 'Precio de entrada', value: fmt2(pos.entry_price) + ' USDT' },
    { label: 'Cantidad BTC', value: fmtN(pos.quantity, 8) },
    { label: 'Costo de entrada', value: fmt2(pos.entry_cost) + ' USDT' },
    { label: 'Stop inicial', value: fmt2(pos.stop_loss) + ' USDT' },
    { label: 'Trailing stop actual', value: pos.trailing_stop_price ? fmt2(pos.trailing_stop_price) + ' USDT' : '—' },
    { label: 'Highest high desde entrada', value: pos.highest_high_since_entry ? fmt2(pos.highest_high_since_entry) + ' USDT' : '—' },
    { label: 'Stop dinámico (eval)', value: pos.current_trailing_stop ? fmt2(pos.current_trailing_stop) + ' USDT' : '—' },
    { label: 'Apertura (UTC)', value: pos.opened_at },
    { label: 'Apertura (local)', value: localTime(pos.opened_at) },
    { label: 'Duración aprox.', value: pos.duration_hours + ' horas' },
    { label: 'Estado', value: pos.status },
    { label: 'Símbolo', value: pos.symbol },
  ];

  const card = document.createElement('div');
  card.className = 'position-card';
  fields.forEach(({ label, value }) => {
    const field = document.createElement('div');
    const lbl = document.createElement('div');
    lbl.className = 'pos-field-label';
    lbl.textContent = label;
    const val = document.createElement('div');
    val.className = 'pos-field-value';
    val.textContent = value;
    field.appendChild(lbl);
    field.appendChild(val);
    card.appendChild(field);
  });
  container.appendChild(card);
}

/* signal history table */
function filterSignals(value) {
  _signalFilter = value;
  renderSignalTable(_filterSignals(_allSignals, value));
}

function _filterSignals(signals, filter) {
  if (!filter) return signals;
  return signals.filter(s => s.signal === filter);
}

function renderSignalTable(signals) {
  const tbody = document.getElementById('signals-tbody');
  tbody.textContent = '';
  if (!signals || signals.length === 0) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 9;
    td.className = 'empty-cell';
    td.textContent = 'Sin evaluaciones.';
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  signals.forEach(ev => {
    const tr = document.createElement('tr');
    const cells = [
      ev.candle_close_time,
      localTime(ev.candle_close_time),
      null, // signal badge handled below
      (ev.reasons || []).join(', ') || '—',
      ev.raw_market_price ? fmt2(ev.raw_market_price) : '—',
      ev.entry_donchian_level ? fmt2(ev.entry_donchian_level) : '—',
      ev.exit_donchian_level ? fmt2(ev.exit_donchian_level) : '—',
      ev.atr ? fmtN(ev.atr, 2) : '—',
      fmt2(ev.equity),
    ];
    cells.forEach((val, i) => {
      const td = document.createElement('td');
      if (i === 2) {
        const span = document.createElement('span');
        span.className = 'sig sig-' + ev.signal;
        span.textContent = ev.signal;
        td.appendChild(span);
      } else {
        td.textContent = val;
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

/* trade history table */
function renderTradeTable(trades) {
  const emptyEl = document.getElementById('trades-empty');
  const wrapper = document.getElementById('trades-table-wrapper');
  const tbody = document.getElementById('trades-tbody');
  tbody.textContent = '';

  if (!trades || trades.length === 0) {
    emptyEl.style.display = '';
    wrapper.style.display = 'none';
    return;
  }
  emptyEl.style.display = 'none';
  wrapper.style.display = '';

  trades.forEach((t, idx) => {
    const tr = document.createElement('tr');
    const isWin = parseFloat(t.net_pnl) > 0;
    const pnlClass = isWin ? 'pnl-pos' : 'pnl-neg';

    const cells = [
      { text: String(t.id) },
      { text: t.opened_at },
      { text: t.closed_at },
      { text: fmt2(t.entry_price) },
      { text: fmt2(t.exit_price) },
      { text: fmtN(t.quantity, 8) },
      { text: fmt2(t.gross_pnl), cls: pnlClass },
      { text: fmt2(t.commission) },
      { text: fmt2(t.net_pnl), cls: pnlClass },
      { text: t.return_pct + '%', cls: pnlClass },
      { text: t.exit_reason },
      { text: t.duration_hours + 'h' },
    ];
    cells.forEach(({ text, cls }) => {
      const td = document.createElement('td');
      td.textContent = text;
      if (cls) td.className = cls;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

/* equity chart */
function renderEquityChart(equityCurve, initialCapitalStr) {
  const container = document.getElementById('equity-chart-container');
  container.textContent = '';

  if (!equityCurve || equityCurve.length < 2) {
    const p = document.createElement('p');
    p.className = 'empty-state';
    p.textContent = 'Sin suficientes puntos para mostrar la curva de equity.';
    container.appendChild(p);
    return;
  }

  const W = 900, H = 280;
  const pad = { top: 20, right: 20, bottom: 48, left: 82 };
  const innerW = W - pad.left - pad.right;
  const innerH = H - pad.top - pad.bottom;

  const equity = equityCurve.map(p => parseFloat(p.equity));
  const ts = equityCurve.map(p => new Date(p.candle_close_time.replace(' ', 'T') + 'Z').getTime());
  const initialCapital = initialCapitalStr ? parseFloat(initialCapitalStr) : null;

  const allVals = initialCapital !== null ? [initialCapital, ...equity] : equity;
  const minE = Math.min(...allVals) * 0.998;
  const maxE = Math.max(...allVals) * 1.002;
  const minT = ts[0], maxT = ts[ts.length - 1];
  const rangeT = maxT - minT || 1;
  const rangeE = maxE - minE || 1;

  const sx = t => pad.left + ((t - minT) / rangeT) * innerW;
  const sy = v => pad.top + (1 - (v - minE) / rangeE) * innerH;

  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Curva de equity a lo largo del tiempo');
  svg.style.width = '100%';
  svg.style.height = 'auto';
  svg.style.display = 'block';

  function el(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    return e;
  }

  // Background
  svg.appendChild(el('rect', { width: W, height: H, fill: '#0d1b2e' }));

  // Grid lines
  [0, 0.25, 0.5, 0.75, 1].forEach(frac => {
    const y = pad.top + frac * innerH;
    svg.appendChild(el('line', { x1: pad.left, y1: y, x2: W - pad.right, y2: y, stroke: '#1e293b', 'stroke-width': 1 }));
  });

  // Reference line: initial capital
  if (initialCapital !== null) {
    const ry = sy(initialCapital);
    svg.appendChild(el('line', {
      x1: pad.left, y1: ry, x2: W - pad.right, y2: ry,
      stroke: '#475569', 'stroke-width': 1, 'stroke-dasharray': '5 4'
    }));
    const refTxt = document.createElementNS(NS, 'text');
    refTxt.setAttribute('x', pad.left - 6);
    refTxt.setAttribute('y', ry + 4);
    refTxt.setAttribute('fill', '#64748b');
    refTxt.setAttribute('font-size', '10');
    refTxt.setAttribute('text-anchor', 'end');
    refTxt.setAttribute('font-family', 'monospace');
    refTxt.textContent = fmt2Compact(initialCapital);
    svg.appendChild(refTxt);
  }

  // Build path
  let d = '';
  ts.forEach((t, i) => {
    const x = sx(t), y = sy(equity[i]);
    d += i === 0 ? `M${x},${y}` : ` L${x},${y}`;
  });

  // Area fill
  const lastX = sx(ts[ts.length - 1]);
  const firstX = sx(ts[0]);
  const bottomY = H - pad.bottom;
  svg.appendChild(el('path', {
    d: d + ` L${lastX},${bottomY} L${firstX},${bottomY} Z`,
    fill: '#1d4ed8', opacity: '0.12'
  }));

  // Equity line
  svg.appendChild(el('path', {
    d, fill: 'none', stroke: '#3b82f6', 'stroke-width': '2', 'stroke-linejoin': 'round'
  }));

  // Y-axis labels (4 ticks)
  [0, 1/3, 2/3, 1].forEach(frac => {
    const v = minE + frac * rangeE;
    const y = sy(v);
    svg.appendChild(el('line', { x1: pad.left - 4, y1: y, x2: pad.left, y2: y, stroke: '#475569', 'stroke-width': 1 }));
    const txt = document.createElementNS(NS, 'text');
    txt.setAttribute('x', pad.left - 8);
    txt.setAttribute('y', y + 4);
    txt.setAttribute('fill', '#94a3b8');
    txt.setAttribute('font-size', '10');
    txt.setAttribute('text-anchor', 'end');
    txt.setAttribute('font-family', 'monospace');
    txt.textContent = fmt2Compact(v);
    svg.appendChild(txt);
  });

  // X-axis line
  svg.appendChild(el('line', { x1: pad.left, y1: H - pad.bottom, x2: W - pad.right, y2: H - pad.bottom, stroke: '#334155', 'stroke-width': 1 }));

  // X-axis ticks (up to 5)
  const numTicks = Math.min(5, ts.length);
  for (let i = 0; i < numTicks; i++) {
    const idx = Math.round(i * (ts.length - 1) / (numTicks - 1));
    const x = sx(ts[idx]);
    svg.appendChild(el('line', { x1: x, y1: H - pad.bottom, x2: x, y2: H - pad.bottom + 4, stroke: '#475569', 'stroke-width': 1 }));
    const txt = document.createElementNS(NS, 'text');
    txt.setAttribute('x', x);
    txt.setAttribute('y', H - pad.bottom + 16);
    txt.setAttribute('fill', '#94a3b8');
    txt.setAttribute('font-size', '9');
    txt.setAttribute('text-anchor', 'middle');
    txt.textContent = fmtDateShort(new Date(ts[idx]));
    svg.appendChild(txt);
  }

  // Tooltip elements
  const vLine = el('line', { x1: 0, x2: 0, y1: pad.top, y2: H - pad.bottom, stroke: '#64748b', 'stroke-width': 1, 'stroke-dasharray': '3 3' });
  vLine.style.display = 'none';
  svg.appendChild(vLine);

  const dot = el('circle', { r: 4, fill: '#3b82f6', stroke: '#e2e8f0', 'stroke-width': 1.5 });
  dot.style.display = 'none';
  svg.appendChild(dot);

  // Mouse overlay
  const overlay = el('rect', { x: pad.left, y: pad.top, width: innerW, height: innerH, fill: 'transparent' });
  overlay.style.cursor = 'crosshair';
  svg.appendChild(overlay);

  const tooltip = document.createElement('div');
  tooltip.className = 'chart-tooltip';
  tooltip.style.display = 'none';

  const wrapper = document.createElement('div');
  wrapper.className = 'chart-wrapper';
  wrapper.appendChild(svg);
  wrapper.appendChild(tooltip);
  container.appendChild(wrapper);

  overlay.addEventListener('mousemove', e => {
    const rect = svg.getBoundingClientRect();
    const ratioX = W / rect.width;
    const mouseX = (e.clientX - rect.left) * ratioX;
    if (mouseX < pad.left || mouseX > W - pad.right) {
      tooltip.style.display = 'none'; vLine.style.display = 'none'; dot.style.display = 'none'; return;
    }
    const tHover = minT + ((mouseX - pad.left) / innerW) * rangeT;
    let nearest = 0, minDist = Infinity;
    ts.forEach((t, i) => { const dist = Math.abs(t - tHover); if (dist < minDist) { minDist = dist; nearest = i; } });

    const px = sx(ts[nearest]), py = sy(equity[nearest]);
    vLine.setAttribute('x1', px); vLine.setAttribute('x2', px);
    vLine.style.display = '';
    dot.setAttribute('cx', px); dot.setAttribute('cy', py);
    dot.style.display = '';

    const ratioY = H / rect.height;
    const screenX = px / ratioX, screenY = py / ratioY;

    tooltip.textContent = '';
    const d1 = document.createElement('div'); d1.className = 'tooltip-date';
    d1.textContent = new Date(ts[nearest]).toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
    const d2 = document.createElement('div'); d2.className = 'tooltip-equity';
    d2.textContent = 'Equity: ' + fmt2(equity[nearest]) + ' USDT';
    const d3 = document.createElement('div'); d3.className = 'tooltip-signal';
    d3.textContent = 'Señal: ' + equityCurve[nearest].signal;
    tooltip.appendChild(d1); tooltip.appendChild(d2); tooltip.appendChild(d3);

    // Keep tooltip inside chart
    const ttW = 190, ttH = 70;
    let tx = screenX + 12, ty = screenY - ttH - 10;
    if (tx + ttW > rect.width) tx = screenX - ttW - 12;
    if (ty < 0) ty = screenY + 10;
    tooltip.style.left = tx + 'px';
    tooltip.style.top = ty + 'px';
    tooltip.style.display = '';
  });

  overlay.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none'; vLine.style.display = 'none'; dot.style.display = 'none';
  });
}

/* frozen config */
function renderFrozenConfig(cfg, launch) {
  const grid = document.getElementById('frozen-config-grid');
  grid.textContent = '';

  const displayMap = [
    ['Símbolo', cfg.symbol],
    ['Timeframe', cfg.trading_interval],
    ['Fuente', cfg.source_interval + ' (agregado a ' + cfg.trading_interval + ')'],
    ['Donchian entrada (lookback)', cfg.entry_lookback],
    ['Donchian salida (lookback)', cfg.exit_lookback],
    ['ATR período', cfg.atr_period],
    ['ATR multiplicador', cfg.atr_multiplier],
    ['EMA período', cfg.ema_period],
    ['EMA slope lookback', cfg.ema_slope_lookback],
    ['Asignación por trade', cfg.allocation_pct + '%'],
    ['Fee', cfg.fee_percentage + '%'],
    ['Slippage', cfg.slippage_percentage + '%'],
    ['Solo largo', cfg.long_only ? 'Sí' : 'No'],
    ['Apalancamiento', cfg.leverage ? 'Sí' : 'No'],
    ['Pyramiding', cfg.pyramiding ? 'Sí' : 'No'],
    ['Stop', cfg.stop_type],
    ['Señal de salida', cfg.exit_rule],
    ['Timing ejecución', cfg.execution_timing],
    ['Estrategia', cfg.strategy_name],
    ['Versión', cfg.strategy_version],
  ];

  if (launch) {
    displayMap.push(['Launch timestamp', launch.launch_timestamp]);
    displayMap.push(['Hash config', launch.frozen_config_hash ? launch.frozen_config_hash.slice(0, 20) + '…' : '—']);
    displayMap.push(['Commit hash', launch.code_commit_hash ? launch.code_commit_hash.slice(0, 12) + '…' : '—']);
  }

  displayMap.forEach(([key, val]) => {
    if (val === undefined || val === null) return;
    const item = document.createElement('div');
    item.className = 'config-item';
    const k = document.createElement('div');
    k.className = 'config-key';
    k.textContent = key;
    const v = document.createElement('div');
    v.className = 'config-value';
    v.textContent = String(val);
    item.appendChild(k);
    item.appendChild(v);
    grid.appendChild(item);
  });
}

/* system status */
function renderSystemStatus(d) {
  const grid = document.getElementById('system-status-grid');
  grid.textContent = '';
  const sys = d.system || {};
  const launch = d.launch;

  const items = [
    { label: 'Launch ID', value: launch ? String(launch.launch_id) : '—', cls: '' },
    { label: 'Launch status', value: launch ? launch.status : '—', cls: launch && launch.status === 'ACTIVE' ? 'status-ok' : (launch && launch.status === 'ERROR' ? 'status-err' : 'status-warn') },
    { label: 'API disponible', value: sys.api_available ? 'Sí' : 'No', cls: sys.api_available ? 'status-ok' : 'status-err' },
    { label: 'Total evaluaciones', value: String(sys.total_evaluations || 0), cls: '' },
    { label: 'Errores de datos', value: String(sys.data_gap_errors || 0), cls: sys.data_gap_errors > 0 ? 'status-warn' : 'status-ok' },
    { label: 'Última vela evaluada (UTC)', value: sys.last_evaluated_candle_close || '—', cls: '' },
    { label: 'Próxima evaluación (UTC)', value: d.next_eligible_close_utc || '—', cls: '' },
    { label: 'Próxima evaluación (local)', value: d.next_eligible_close_utc ? localTime(d.next_eligible_close_utc) : '—', cls: '' },
    { label: 'Datos generados a (UTC)', value: d.generated_at || '—', cls: '' },
  ];

  if (launch) {
    items.push({ label: 'Capital inicial', value: fmt2(launch.initial_capital) + ' USDT', cls: '' });
    items.push({ label: 'Símbolo', value: launch.symbol || '—', cls: '' });
    items.push({ label: 'Estrategia', value: launch.strategy_name || '—', cls: '' });
    items.push({ label: 'Versión', value: launch.strategy_version || '—', cls: '' });
  }

  items.forEach(({ label, value, cls }) => {
    const item = document.createElement('div');
    item.className = 'status-item';
    const lbl = document.createElement('div');
    lbl.className = 'status-label';
    lbl.textContent = label;
    const val = document.createElement('div');
    val.className = 'status-value' + (cls ? ' ' + cls : '');
    val.textContent = value;
    item.appendChild(lbl);
    item.appendChild(val);
    grid.appendChild(item);
  });
}

/* ------------------------------------------------------------ UI helpers */

function setUpdating(on) {
  const el = document.getElementById('update-indicator');
  el.textContent = on ? 'Actualizando…' : '';
}

function updateLastRefreshed() {
  const el = document.getElementById('last-updated');
  el.textContent = 'Actualizado: ' + new Date().toLocaleTimeString();
}

function showFetchError(msg) {
  const el = document.getElementById('error-banner');
  el.textContent = 'No se pudo actualizar. Mostrando los últimos datos disponibles. (' + msg + ')';
  el.style.display = '';
}

function clearFetchError() {
  const el = document.getElementById('error-banner');
  el.textContent = '';
  el.style.display = 'none';
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

/* ------------------------------------------------------------ formatting */

function fmt2(val) {
  const n = parseFloat(val);
  if (isNaN(n)) return val || '—';
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmt2Compact(n) {
  if (n >= 10000) return (n / 1000).toFixed(1) + 'k';
  return n.toFixed(0);
}

function fmt8(val) {
  const n = parseFloat(val);
  if (isNaN(n)) return val || '—';
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtN(val, dp) {
  const n = parseFloat(val);
  if (isNaN(n)) return val || '—';
  return n.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
}

function fmtDateShort(date) {
  if (isNaN(date.getTime())) return '?';
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function localTime(isoStr) {
  if (!isoStr) return '—';
  const d = new Date(isoStr.includes('T') ? isoStr + (isoStr.endsWith('Z') ? '' : 'Z') : isoStr.replace(' ', 'T') + 'Z');
  if (isNaN(d.getTime())) return isoStr;
  try {
    return new Intl.DateTimeFormat(undefined, {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
      timeZoneName: 'short'
    }).format(d);
  } catch {
    return d.toLocaleString();
  }
}

function signalColor(sig) {
  return { WAIT: '', BUY_PENDING: 'yellow', LONG: 'green', SELL_PENDING: 'orange', EXITED: 'blue', ERROR_DATA_GAP: 'red' }[sig] || '';
}

function pnlColor(val) {
  const n = parseFloat(val);
  if (isNaN(n) || n === 0) return '';
  return n > 0 ? 'green' : 'red';
}

/* ------------------------------------------------------------ exports */

function exportSignals()  { window.location.href = API + '/signals/export'; }
function exportTrades()   { window.location.href = API + '/trades/export'; }
function exportEquity()   { window.location.href = API + '/equity/export'; }
function exportManifest() { window.location.href = API + '/config/export'; }
function exportDiagnostic() { window.location.href = API + '/diagnostic'; }

/* --------------------------------------------------------- health fetch */

async function fetchHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) return;
    const data = await res.json();
    renderHealthStatus(data);
  } catch (_) {
    // Health fetch failure is non-fatal
  }
}

function renderHealthStatus(h) {
  const container = document.getElementById('health-container');
  if (!container) return;
  container.textContent = '';

  const statusColors = { HEALTHY: 'status-ok', DEGRADED: 'status-warn', ERROR: 'status-err' };
  const cls = statusColors[h.status] || '';

  const badge = document.createElement('div');
  badge.className = 'health-badge ' + cls;
  badge.textContent = h.status;
  container.appendChild(badge);

  const grid = document.createElement('div');
  grid.className = 'status-grid';

  const items = [
    { label: 'Base de datos', value: h.db_status, cls: h.db_status === 'ok' ? 'status-ok' : 'status-err' },
    { label: 'Launch', value: h.launch_status, cls: h.launch_status === 'ACTIVE' ? 'status-ok' : 'status-warn' },
    { label: 'Heartbeat', value: h.heartbeat_status, cls: h.heartbeat_status === 'active' ? 'status-ok' : (h.heartbeat_status === 'idle' ? '' : 'status-warn') },
    { label: 'Datos', value: h.data_freshness, cls: h.data_freshness === 'fresh' ? 'status-ok' : 'status-warn' },
    { label: 'Último ciclo', value: h.last_heartbeat_result || '—', cls: h.last_heartbeat_result === 'OK' ? 'status-ok' : (h.last_heartbeat_result === 'ERROR' ? 'status-err' : '') },
    { label: 'Edad heartbeat', value: h.heartbeat_age_seconds != null ? (h.heartbeat_age_seconds / 60).toFixed(1) + ' min' : '—', cls: '' },
    { label: 'Edad datos (h)', value: h.last_candle_age_hours != null ? h.last_candle_age_hours.toFixed(1) + 'h' : '—', cls: '' },
    { label: 'Verificado (UTC)', value: h.checked_at || '—', cls: '' },
  ];

  items.forEach(({ label, value, cls }) => {
    const item = document.createElement('div');
    item.className = 'status-item';
    const lbl = document.createElement('div');
    lbl.className = 'status-label';
    lbl.textContent = label;
    const val = document.createElement('div');
    val.className = 'status-value' + (cls ? ' ' + cls : '');
    val.textContent = value;
    item.appendChild(lbl);
    item.appendChild(val);
    grid.appendChild(item);
  });
  container.appendChild(grid);

  if (h.issues && h.issues.length > 0) {
    const issueList = document.createElement('ul');
    issueList.className = 'health-issues';
    h.issues.forEach(issue => {
      const li = document.createElement('li');
      li.textContent = issue;
      issueList.appendChild(li);
    });
    container.appendChild(issueList);
  }
}

/* --------------------------------------------------------- events fetch */

async function fetchEvents() {
  try {
    const res = await fetch(API + '/events/unread');
    if (!res.ok) return;
    const events = await res.json();
    renderEvents(events);
    _notifyNewEvents(events);
  } catch (_) {
    // Events fetch failure is non-fatal
  }
}

function renderEvents(events) {
  const container = document.getElementById('events-container');
  if (!container) return;
  container.textContent = '';

  if (!events || events.length === 0) {
    const p = document.createElement('p');
    p.className = 'empty-state';
    p.textContent = 'No hay eventos sin leer.';
    container.appendChild(p);
    return;
  }

  const list = document.createElement('div');
  list.className = 'events-list';

  events.forEach(ev => {
    const item = document.createElement('div');
    item.className = 'event-item sev-' + (ev.severity || 'INFO').toLowerCase();

    const header = document.createElement('div');
    header.className = 'event-header';

    const type = document.createElement('span');
    type.className = 'event-type';
    type.textContent = ev.event_type;

    const ts = document.createElement('span');
    ts.className = 'event-ts';
    ts.textContent = ev.timestamp_utc;

    const btn = document.createElement('button');
    btn.className = 'btn-secondary btn-xs';
    btn.type = 'button';
    btn.textContent = 'Marcar leído';
    btn.addEventListener('click', () => markEventRead(ev.id));

    header.appendChild(type);
    header.appendChild(ts);
    header.appendChild(btn);

    const msg = document.createElement('div');
    msg.className = 'event-msg';
    msg.textContent = ev.message;

    item.appendChild(header);
    item.appendChild(msg);
    list.appendChild(item);
  });

  container.appendChild(list);
}

async function markEventRead(eventId) {
  try {
    await fetch(API + '/events/' + eventId + '/read', { method: 'POST' });
    fetchEvents();
  } catch (_) {}
}

/* ------------------------------------------------ browser notifications */

function requestNotificationPermission() {
  if (!('Notification' in window)) {
    alert('Tu navegador no soporta notificaciones.');
    return;
  }
  Notification.requestPermission().then(perm => {
    const btn = document.getElementById('btn-notifications');
    if (btn) {
      btn.textContent = perm === 'granted' ? 'Notificaciones activas' : 'Notificaciones bloqueadas';
    }
  });
}

function _notifyNewEvents(events) {
  if (!('Notification' in window) || Notification.permission !== 'granted') return;
  events.forEach(ev => {
    if (ev.id > _lastSeenEventId) {
      _lastSeenEventId = ev.id;
      new Notification('Paper Trading — ' + ev.event_type, {  // eslint-disable-line no-new
        body: ev.message,
        tag: 'paper-ev-' + ev.id,
      });
    }
  });
}

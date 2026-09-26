/* Triffy front-end.
 *
 * The map stays calm: only the planned route is coloured, by the traffic
 * expected along it. The road network is loaded for the city's extent and
 * name, not drawn.
 */

const API = '';

// Which world are we showing? 'sim' is Kolkata with modelled conditions; 'live'
// is London Zone 1 driven entirely by real camera measurements. Keeping this an
// explicit, visible switch is a deliberate honesty choice - the user should
// never have to guess whether numbers on screen were measured or modelled.
let MODE = 'sim';
const ENDPOINTS = {
  sim:  { network: '/api/network',      state: '/api/state',      plan: '/api/plan' },
  live: { network: '/api/live/network', state: '/api/live/state', plan: '/api/live/plan' }
};
const ep = k => ENDPOINTS[MODE][k];

let MAP, ROUTE_LAYER, INC_LAYER;
let LAST_STATE = null;
let PLAN = null;
let SELECTED = 0;

/* ---------- colour ---------- */

// Route traffic colours in four clear bands: flowing (the route's own blue),
// slowing, congested, near gridlock. Same values as --blue and --t1..--t3 in
// CSS. Only the route is coloured; the rest of the map stays neutral.
const TRAFFIC = [null, '#e7a300', '#e0600e', '#b42318'];
const ROUTE_BLUE = '#1f5ae6';

/* ---------- boot ---------- */

async function boot() {
  MAP = L.map('map', { zoomControl: false, preferCanvas: true,
                       attributionControl: true });
  L.control.zoom({ position: 'bottomright' }).addTo(MAP);
  document.querySelector('.leaflet-control-zoom-in').innerHTML = '<svg><use href="#i-plus"/></svg>';
  document.querySelector('.leaflet-control-zoom-out').innerHTML = '<svg><use href="#i-minus"/></svg>';
  // Standard OSM raster tiles need no API key. A CSS filter on the tile pane
  // turns them into a quiet grey base (see style.css): our own coloured road
  // network is the information layer; the basemap only supplies context.
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors | Triffy prototype'
  }).addTo(MAP);

  ROUTE_LAYER = L.layerGroup().addTo(MAP);
  INC_LAYER = L.layerGroup().addTo(MAP);

  // On a small laptop three columns leave the map a sliver, and on a phone the
  // key would cover the map, so it starts closed there; the chip button opens it.
  if (innerWidth <= 1280) {
    document.body.classList.add('drawer-off');
    const t = document.getElementById('drawerToggle');
    t.setAttribute('aria-expanded', 'false');
    t.setAttribute('aria-label', 'Show the map key');
  }
  addEventListener('resize', () => syncStatusHeight());

  // /?mode=live opens straight into London (bookmarkable).
  const PARAMS = new URLSearchParams(location.search);
  if (PARAMS.get('mode') === 'live') MODE = 'live';
  await loadCity();

  await loadProfiles();
  await loadPlaces();
  wire();
  // A trip in the query string, so the chat can hand someone a link back to
  // this map with their route already on it: /?from=A&to=B&at=18:30
  // The inputs are the single source of truth for plan(), so filling them in
  // is all this needs - no second planning path to drift out of step.
  applyTripParams(PARAMS);
  plan();
  // Keep the clock and the traffic figures current while the page is open
  // (and not while it sits in a background tab).
  setInterval(() => { if (!document.hidden) refresh(); }, 30000);
}

/* ---------- a trip passed in the URL ---------- */

// /?from=Park+Circus&to=Esplanade&at=18:30 — the link the chat bot sends.
// Values go through the same inputs a person would type into, so they get the
// same geocoding, the same validation and the same errors. Anything missing is
// simply left at its default.
function applyTripParams(params) {
  const fill = (id, value) => {
    const el = document.getElementById(id);
    if (el && value) el.value = value;
  };
  fill('origin', params.get('from'));
  fill('dest', params.get('to'));
  fill('depart', params.get('at'));
  fill('arriveBy', params.get('by'));
  const user = params.get('user');
  if (user) {
    // Only select a persona that actually exists, or the <select> silently
    // ends up blank and every route is planned for nobody.
    const sel = document.getElementById('persona');
    if (sel && [...sel.options].some(o => o.value === user)) sel.value = user;
  }
}

/* ---------- load (or reload) the active city ---------- */

async function loadCity() {
  ROUTE_LAYER.clearLayers();
  INC_LAYER.clearLayers();
  PLAN = null;

  const net = await (await fetch(API + ep('network'))).json();
  document.getElementById('cityLabel').textContent = net.city || '';

  const b = net.bbox;
  MAP.fitBounds([[b[0], b[1]], [b[2], b[3]]]);

  applyModeChrome();
  await refresh();
}

// Which world this is, stated where the task starts. The honesty mechanism of
// the whole product: never let a user guess. In London it also follows the
// data: with no recent camera readings it must not claim live conditions.
function setProvenance(live, hasLiveData) {
  let html;
  if (!live) {
    html = `<svg><use href="#i-info"/></svg><span><b>Simulated traffic, real roads.</b>
       <span class="prov-more">Kolkata's road network is real; its traffic is
       modelled, because Kolkata's camera feeds are not public.</span></span>`;
  } else if (hasLiveData) {
    html = `<svg><use href="#i-camera"/></svg><span><b>Live, measured conditions.</b>
       <span class="prov-more">Real London roads, driven only by Transport for
       London cameras that we measure ourselves. No simulator. Powered by TfL
       Open Data.</span></span>`;
  } else {
    html = `<svg><use href="#i-info"/></svg><span><b>No live data right now.</b>
       <span class="prov-more">London's roads are real, but no recent camera
       readings are coming in, so traffic is estimated from typical conditions
       for this time of day.</span></span>`;
  }
  document.getElementById('provenance').innerHTML = html;
}

function applyModeChrome() {
  // Simulation-only controls have no meaning against live data: you cannot
  // fast-forward London, and you certainly cannot inject a real collision.
  const live = MODE === 'live';
  document.body.dataset.mode = MODE;
  document.querySelectorAll('[data-mode]').forEach(b => {
    if (b.tagName !== 'BUTTON') return;
    const on = b.dataset.mode === MODE;
    b.classList.toggle('on', on);
    b.setAttribute('aria-pressed', String(on));
  });

  setProvenance(live, true);

  document.getElementById('clockLabel').textContent = live ? 'London' : 'Kolkata';
  document.getElementById('arriveBy').parentElement.style.display = live ? 'none' : '';
  document.getElementById('depart').parentElement.style.display = live ? 'none' : '';
  document.querySelector('.opts').style.display = live ? 'none' : '';

  const o = document.getElementById('origin'), d = document.getElementById('dest');
  if (live) { o.value = "King's Cross"; d.value = 'Tower Bridge'; }
  else      { o.value = 'Park Circus';   d.value = 'BBD Bagh'; }
  loadPlaces();
}

const PLACES = {
  live: ["King's Cross", 'Tower Bridge', 'Piccadilly Circus', 'Oxford Circus',
    'Trafalgar Square', 'Waterloo', 'Euston', 'Paddington', 'Victoria',
    'Liverpool Street', 'Bank', 'Angel Islington', 'Camden Town', 'Shoreditch',
    'Elephant and Castle', 'Vauxhall Cross', 'Hyde Park Corner', 'Marble Arch',
    'Westminster', 'Whitechapel', 'Farringdon', 'Holborn', 'Knightsbridge',
    'Chelsea Bridge', 'Borough', 'Aldgate', 'Soho', 'Mayfair', 'Oxford Street'],
  sim: []
};

async function loadPlaces() {
  if (MODE === 'live') {
    document.getElementById('places').innerHTML =
      PLACES.live.map(n => `<option value="${n}">`).join('');
    return;
  }
  // Populate the datalist from landmark names the geocoder knows.
  const names = ['Park Circus','BBD Bagh','Esplanade','Sealdah','Park Street',
    'Victoria Memorial','Howrah Bridge approach','College Street','New Market',
    'Alipore','Bhowanipore','Rabindra Sadan','Girish Park','Moulali','Rajabazar',
    'Shakespeare Sarani','Camac Street','Chandni Chowk','Bowbazar','Eden Gardens',
    'Fort William','Exide Crossing','Strand Road','Maa Flyover','Mullick Bazar'];
  document.getElementById('places').innerHTML =
    names.map(n => `<option value="${n}">`).join('');
}

// How the built-in travel profiles are offered to a user. Unknown profiles
// fall back to the name the server gives them.
const PROFILE_LABELS = {
  exec: 'Car · must arrive on time',
  rider: 'Motorcycle · fastest',
  student: 'Auto-rickshaw · cheapest',
  cabbie: 'Taxi'
};

async function loadProfiles() {
  const p = await (await fetch(API + '/api/profiles')).json();
  const sel = document.getElementById('persona');
  sel.innerHTML = Object.entries(p)
    .map(([k, v]) => `<option value="${k}">${PROFILE_LABELS[k] || v.name}</option>`).join('');
  sel.value = 'exec';
  window.PROFILES = p;
  hintPersona();
}

function hintPersona() {
  const k = document.getElementById('persona').value;
  const p = window.PROFILES?.[k];
  if (!p) { document.getElementById('personaHint').textContent = ''; return; }
  let style = 'chases the fastest route';
  if (p.risk_aversion > 1.2) style = 'plays it safe';
  else if (p.risk_aversion > 0.5) style = 'balances speed and safety';
  const tripWord = p.trips === 1 ? 'trip' : 'trips';
  const trips = p.trips ? `${p.trips} ${tripWord} learned from` : 'no trips learned from yet';
  const vehicle = p.vehicle.charAt(0).toUpperCase() + p.vehicle.slice(1);
  document.getElementById('personaHint').textContent = `${vehicle} · ${style} · ${trips}`;
}

/* ---------- live state ---------- */

async function refresh() {
  const s = await (await fetch(API + ep('state'))).json();
  LAST_STATE = s;
  // The live server reports its own (Indian) wall clock; the chip is labelled
  // London, so it must show London's time, like the route times do.
  document.getElementById('clock').textContent =
    MODE === 'live' ? clockAt(Date.now() / 1000) : s.clock;
  // Kolkata's clock is real unless the server pins it (TRIFFY_SIM_CLOCK), and
  // the label must say which, or a pinned 18:30 reads as a wrong "now".
  if (MODE === 'sim') {
    document.getElementById('clockLabel').textContent =
      s.real_time === false ? 'Simulated time' : 'Kolkata';
  }

  if (MODE === 'sim') renderIncidents(s.incidents);
  renderKpis(s.stats);
}

function renderKpis(st) {
  const chip = (v, k, cls = '') => `<span class="chip ${cls}"><b>${v}</b>${k}</span>`;
  if (MODE === 'live') {
    const age = st.data_age_s;
    setProvenance(true, age != null);
    document.getElementById('kpis').innerHTML =
      (age == null
        ? chip('No live data', '')
        : `<span class="chip live">Live</span>` + chip(ageShort(age), 'ago')) +
      chip(`${st.arterial_kph}`, 'km/h on main roads');
    syncStatusHeight();
    return;
  }
  document.getElementById('kpis').innerHTML =
    chip(`${st.arterial_kph}`, 'km/h on main roads') +
    chip(`${st.congested_pct}%`, 'of roads congested');
  syncStatusHeight();
}

// The drawer hangs below the status chips; on a narrow laptop they wrap to two
// rows, and the drawer must move down with them rather than hide under them.
function syncStatusHeight() {
  const h = document.querySelector('.status').getBoundingClientRect().height;
  document.documentElement.style.setProperty('--status-h', Math.round(h) + 'px');
}

function renderIncidents(list) {
  INC_LAYER.clearLayers();
  list.forEach(i => {
    L.circleMarker([i.lat, i.lon], {
      radius: 8 + 5 * i.intensity, color: '#c0341d', weight: 2,
      fillColor: '#c0341d', fillOpacity: 0.22
    }).bindPopup(`<b>${i.label}</b><br>Since ${i.started}<br>` +
                 `Clears about ${i.clears}`).addTo(INC_LAYER);
  });
}

/* ---------- planning ---------- */

async function plan() {
  const origin = document.getElementById('origin').value;
  const destination = document.getElementById('dest').value;
  const depart = document.getElementById('depart').value.trim();
  const arriveBy = document.getElementById('arriveBy').value.trim();
  const user = document.getElementById('persona').value;
  const results = document.getElementById('results');
  results.innerHTML = `
    <div class="answer" aria-busy="true">
      <div class="skel" style="height:46px;width:42%"></div>
      <div class="skel" style="height:14px;width:72%;margin-top:14px"></div>
      <div class="skel" style="height:8px;margin-top:20px"></div>
      <div class="skel" style="height:22px;width:60%;margin-top:16px"></div>
    </div>`;

  try {
    if (arriveBy && MODE === 'sim') {
      await planLeaveBy(origin, destination, arriveBy, user);
      return;
    }
    const body = MODE === 'live'
      ? { origin, destination, user, k: 3 }
      : { origin, destination, depart: depart || null, user, k: 3 };
    const p = await post(ep('plan'), body);
    PLAN = p;
    SELECTED = 0;
    showAdvisory(MODE === 'live' ? liveAdvisory(p) : p.advisory);
    setCompact(true);
    renderRoutes();
  } catch (err) {
    showError(err.message);
  }
}

// Kolkata's "arrive by" question: when to leave, answered on the 90th percentile.
async function planLeaveBy(origin, destination, arriveBy, user) {
  const lb = await post('/api/leaveby',
    { origin, destination, arrive_by: arriveBy, user });
  if (!lb.ok) { showError(lb.detail); return; }
  showAdvisory(`Leave by <b>${lb.leave_at}</b> to reach ${destination} by ` +
    `${lb.deadline} with ${Math.round(lb.confidence * 100)}% confidence. ` +
    `Typical run ${lb.typical_min} min; budget ${lb.buffered_min} min.`);
  PLAN = { routes: [lb.route] };
  SELECTED = 0;
  setCompact(true);
  renderRoutes();
}

// Say how much of THIS route the cameras actually inform. "Planned against
// live conditions" is a claim, and a route can be 98% or 2% camera-informed
// depending on whether it runs along watched corridors. Showing the number
// makes the claim checkable rather than asserted.
function liveAdvisory(p) {
  // No fresh camera readings (no collector running): say that plainly rather
  // than "live conditions from 0 cameras".
  if (!p.cameras_reporting) {
    return 'No live camera readings right now, so this route uses ' +
      '<b>typical conditions</b> for this time of day.';
  }
  const age = p.data_age_s;
  const ls = p.routes[0]?.live_share;
  let cover = '';
  if (ls) {
    const pct = ls.informed_pct;
    cover = ` Cameras inform <b>${pct}%</b> of this route by distance` +
            (pct < 50 ? ' — the rest falls back to typical conditions for this time of day.' : '.');
  }
  return 'Planned against <b>live measured conditions</b> from ' +
    `<b>${p.cameras_reporting}</b> real cameras` +
    (age == null ? '.' : `; newest reading ${ageShort(age)} old.`) + cover;
}

// After routing, the form folds to one line (tap to edit), as in any maps app.
function setCompact(on) {
  const search = document.getElementById('search');
  const sum = document.getElementById('odSummary');
  search.classList.toggle('compact', on);
  document.body.classList.toggle('routed', on);
  sum.hidden = !on;
  // On a phone: an answer opens the sheet to show it; editing opens it fully.
  setSheet(on ? 'half' : 'full');
  if (on) {
    const v = id => document.getElementById(id).value.trim();
    const who = document.getElementById('persona');
    const whoName = who.options[who.selectedIndex] ? who.options[who.selectedIndex].text : '';
    let when = 'leave now';
    if (MODE === 'sim' && v('arriveBy')) when = 'arrive by ' + v('arriveBy');
    else if (MODE === 'sim' && v('depart')) when = 'leave at ' + v('depart');
    document.getElementById('odSumText').innerHTML =
      `${v('origin')} → ${v('dest')}<small>${whoName} · ${when}</small>`;
  }
}

function showError(msg) {
  setCompact(false);
  // Take the previous route off the map too, or it reads as the answer to
  // the search that just failed.
  PLAN = null;
  ROUTE_LAYER.clearLayers();
  document.getElementById('advisory').classList.add('hidden');
  document.getElementById('results').innerHTML =
    `<div class="error"><svg><use href="#i-alert"/></svg><span>${msg}</span></div>`;
}

function showAdvisory(text) {
  const el = document.getElementById('advisory');
  if (!text) { el.classList.add('hidden'); return; }
  el.innerHTML = `<svg><use href="#i-info"/></svg><span>${text}</span>`;
  el.classList.remove('hidden');
}

/* ---------- the answer ---------- */

const mins = s => Math.round(s / 60);

// Kolkata runs on network time (seconds since midnight); London plans in real
// Unix time and must be shown in London's own clock, not this laptop's.
function clockAt(secs) {
  if (MODE === 'live') {
    return new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit',
      timeZone: 'Europe/London' }).format(new Date(secs * 1000));
  }
  const t = ((Math.round(secs) % 86400) + 86400) % 86400;
  return String(Math.floor(t / 3600)).padStart(2, '0') + ':' +
         String(Math.floor((t % 3600) / 60)).padStart(2, '0');
}

function fmtDist(m) {
  return m >= 1000 ? (m / 1000).toFixed(1) + ' km' : Math.round(m / 10) * 10 + ' m';
}

function relOf(r) {
  const rel = Math.round((r.reliability || 0) * 100);
  let cls = 'poor';
  if (rel > 88) cls = 'good';
  else if (rel > 76) cls = 'ok';
  return { rel, cls };
}

// "45 s" or "4 min": how old a reading is, short enough for a chip.
function ageShort(s) {
  return s < 90 ? s + ' s' : Math.round(s / 60) + ' min';
}

// True once a plan with at least one route is on screen.
function hasRoutes() {
  return Boolean(PLAN?.routes?.length);
}

function stepIcon(instruction) {
  const s = instruction.toLowerCase();
  if (s.startsWith('start')) return 'start';
  if (s.startsWith('arrive')) return 'arrive';
  if (s.includes('bear left') || s.includes('slight left')) return 'bear-left';
  if (s.includes('bear right') || s.includes('slight right')) return 'bear-right';
  if (s.includes('left')) return 'left';
  if (s.includes('right')) return 'right';
  return 'straight';
}

function roadChips(roads) {
  return (roads || []).slice(0, 4).map(n => `<span class="road">${n}</span>`)
    .join('<svg class="road-sep"><use href="#i-chev"/></svg>');
}

// Alternatives are named by how they differ from the recommendation, the same
// way for every one of them, rather than "Alternative 1" beside "Shorter, slower".
function routeName(x, i) {
  if (i === 0) return 'Recommended';
  const rec = PLAN.routes[0];
  const dt = mins(x.median_s) - mins(rec.median_s);
  const dd = x.distance_m - rec.distance_m;
  let len = 'Same distance';
  if (dd < -200) len = 'Shorter';
  else if (dd > 200) len = 'Longer';
  let time = 'same time';
  if (dt > 0) time = `${dt} min slower`;
  else if (dt < 0) time = `${-dt} min faster`;
  return `${len}, ${time}`;
}
// What makes a route different: the first main roads it uses that the
// recommendation does not. Identical main roads are said to be identical.
function routeVia(x, i) {
  const roads = x.roads || [];
  if (i === 0) return 'via ' + (roads.slice(0, 2).join(', ') || 'local roads');
  const base = new Set(PLAN.routes[0].roads || []);
  const own = roads.filter(r => !base.has(r));
  return own.length ? 'via ' + own.slice(0, 2).join(', ')
                    : 'same main roads, different side streets';
}

function routeCard(x, i) {
  const q = relOf(x);
  const sel = i === SELECTED;
  return `
      <button type="button" class="route ${sel ? 'sel' : ''}" data-i="${i}" aria-pressed="${sel}">
        <span class="route-name">${routeName(x, i)}</span>
        <span class="route-min">${mins(x.median_s)} min</span>
        <span class="route-sub"><span class="rel ${q.cls}">${q.rel}%</span> · ${fmtDist(x.distance_m)} · ${routeVia(x, i)}</span>
      </button>`;
}

function stepItem(s) {
  const dist = s.distance_m ? fmtDist(s.distance_m) : '';
  return `
      <li><span class="st-ico"><svg><use href="#i-step-${stepIcon(s.instruction)}"/></svg></span>
        <span>${s.instruction}</span><span class="st-d">${dist}</span></li>`;
}

function renderRoutes() {
  const box = document.getElementById('results');
  if (!hasRoutes()) {
    box.innerHTML = '<p class="hint">No route found between those two places.</p>';
    ROUTE_LAYER.clearLayers();
    return;
  }
  const r = PLAN.routes[SELECTED];
  const { rel, cls } = relOf(r);

  // Range bar: p10 to p90 on an axis with room either side, so a tight
  // spread looks tight and a long tail looks long.
  const lo = r.p10_s, hi = r.p90_s, med = r.median_s;
  const pad = Math.max((hi - lo) * 0.7, 60);
  const a0 = Math.max(0, lo - pad), a1 = hi + pad;
  const pct = x => (100 * (x - a0) / (a1 - a0)).toFixed(2);
  const midInBand = hi > lo ? (100 * (med - lo) / (hi - lo)).toFixed(1) : 50;

  const answer = `
    <div class="answer">
      <div class="answer-top">
        <div class="answer-min">${mins(med)}<small>min</small></div>
        <div class="answer-arrive">Arrive about<b>${clockAt(r.depart_s + med)}</b></div>
      </div>
      <div class="answer-meta">
        <span class="rel ${cls}">${rel}% predictable</span><span class="dot">·</span>${fmtDist(r.distance_m)}<span class="dot">·</span>leaving ${clockAt(r.depart_s)}
      </div>
      <div class="range" aria-label="Likely trip time from ${mins(lo)} to ${mins(hi)} minutes, typically ${mins(med)}">
        <div class="range-track">
          <div class="range-band" style="left:${pct(lo)}%;width:${(pct(hi) - pct(lo)).toFixed(2)}%;--mid:${midInBand}%"></div>
          <div class="range-mid" style="left:${pct(med)}%"></div>
        </div>
        <div class="range-lbl"><span>1-in-10 best <b>${mins(lo)} min</b></span><span>1-in-10 worst <b>${mins(hi)} min</b></span></div>
      </div>
      <div class="roads">${roadChips(r.roads)}</div>
    </div>`;

  const cards = PLAN.routes.map(routeCard).join('');
  const list = PLAN.routes.length > 1
    ? `<h3 class="list-h">${PLAN.routes.length} routes compared</h3>${cards}` : '';

  let steps = '';
  if (r.steps?.length) {
    const items = r.steps.slice(0, 8).map(stepItem).join('');
    const more = r.steps.length > 8
      ? `<p class="steps-more">${r.steps.length - 8} more steps</p>` : '';
    steps = `<h3 class="list-h">Directions</h3><ol class="steps">${items}</ol>${more}`;
  }

  box.innerHTML = answer + list + steps;
  box.querySelectorAll('.route').forEach(el => {
    el.onclick = () => {
      const i = parseInt(el.dataset.i, 10);
      if (i >= 0 && i !== SELECTED) { SELECTED = i; renderRoutes(); }
    };
  });
  drawRoutes();
  setSheet(SHEET);   // the answer's height changed, so the rests did too
}

function mapPadding() {
  // On a phone, frame the route in the map left between the pills and the sheet.
  if (PHONE.matches) {
    const top = document.querySelector('.status').getBoundingClientRect().bottom + 12;
    return { paddingTopLeft: [28, top],
             paddingBottomRight: [28, Math.min(SHEET_SHOWN, innerHeight * 0.7) + 28] };
  }
  const plan = document.querySelector('.plan').getBoundingClientRect();
  const drawerOn = !document.body.classList.contains('drawer-off');
  const right = drawerOn ? document.getElementById('drawer').getBoundingClientRect().width + 36 : 36;
  return { paddingTopLeft: [plan.right + 36, 72], paddingBottomRight: [right, 90] };
}

// Minutes ride on each line, placed at different points along each route so
// the tags do not stack on a stretch the routes share.
function routeTag(r, i) {
  const g = r.geometry;
  const f = i === SELECTED ? 0.5 : 0.3 + 0.2 * i;
  const at = g[Math.min(g.length - 1, Math.floor(g.length * (f % 1)))];
  return L.tooltip({ permanent: true, direction: 'center', interactive: false,
    className: 'route-tag' + (i === SELECTED ? ' sel' : '') })
    .setLatLng(at).setContent(mins(r.median_s) + ' min');
}

function drawRoutes() {
  ROUTE_LAYER.clearLayers();
  if (!hasRoutes()) return;
  // Alternatives first so the selected route draws on top; every route gets a
  // white casing so it reads above the traffic colours, as in any maps app.
  PLAN.routes.forEach((r, i) => {
    if (i === SELECTED) return;
    L.polyline(r.geometry, { color: '#ffffff', weight: 8, opacity: 1, lineCap: 'round' }).addTo(ROUTE_LAYER);
    L.polyline(r.geometry, { color: '#9aa4ae', weight: 5, opacity: 1, lineCap: 'round' })
      .on('click', () => { SELECTED = i; renderRoutes(); }).addTo(ROUTE_LAYER);
    routeTag(r, i).addTo(ROUTE_LAYER);
  });
  const sel = PLAN.routes[SELECTED];
  if (!sel) return;
  L.polyline(sel.geometry, { color: '#ffffff', weight: 11, opacity: 1, lineCap: 'round' }).addTo(ROUTE_LAYER);
  // The chosen route is coloured along its length by the traffic expected when
  // you reach each stretch: blue where it flows, then yellow, orange and red,
  // in the same colours as the roads. An engine that sends no `traffic` gets a
  // plain blue line.
  const runs = sel.traffic || [];
  if (runs.length) {
    runs.forEach(run => L.polyline(run.g, {
      color: run.level ? TRAFFIC[run.level] : ROUTE_BLUE,
      weight: 6.5, opacity: 1, lineCap: 'round'
    }).addTo(ROUTE_LAYER));
  } else {
    L.polyline(sel.geometry, { color: ROUTE_BLUE, weight: 6.5, opacity: 1, lineCap: 'round' }).addTo(ROUTE_LAYER);
  }
  routeTag(sel, SELECTED).addTo(ROUTE_LAYER);
  const g = sel.geometry;
  if (g.length) {
    L.marker(g[0], { interactive: false, keyboard: false, icon: L.divIcon({
      className: '', html: '<div class="pin-start"></div>', iconSize: [16, 16], iconAnchor: [8, 8] }) }).addTo(ROUTE_LAYER);
    L.marker(g[g.length - 1], { interactive: false, keyboard: false, icon: L.divIcon({
      className: 'pin-dest', iconSize: [28, 35], iconAnchor: [14, 34],
      html: '<svg viewBox="0 0 24 30"><path d="M12 29s10-10.5 10-17A10 10 0 0 0 2 12c0 6.5 10 17 10 17z" fill="#1f5ae6" stroke="#fff" stroke-width="2"/><circle cx="12" cy="12" r="3.6" fill="#fff"/></svg>' }) }).addTo(ROUTE_LAYER);
  }
  MAP.fitBounds(L.polyline(sel.geometry).getBounds(), mapPadding());
}

/* ---------- phones: the bottom sheet ---------- */

// On a phone the plan is a sheet over a full-screen map (style.css), resting
// at one of three heights, as in Apple or Google Maps:
//   peek - the trip only, so the map has the screen;
//   half - the trip and the answer;
//   full - everything; only now does the sheet's content scroll.
// The rests come from the content itself, so "half" always ends just below
// the answer, however long the trip summary or advisory above it runs.
const PHONE = matchMedia('(max-width: 860px)');
let SHEET = 'half';          // the rest the sheet is at, or heading for
let SHEET_SHOWN = 0;         // px of sheet on screen, for framing the map

function sheetRests() {
  const sheet = document.querySelector('.plan');
  const top = sheet.getBoundingClientRect().top - sheet.scrollTop;
  const bottomOf = sel => {
    const el = sheet.querySelector(sel);
    return el && el.offsetParent ? el.getBoundingClientRect().bottom - top : 0;
  };
  const full = sheet.offsetHeight;
  const peek = Math.min(Math.max(bottomOf('.search') + 18, 150), innerHeight * 0.42);
  // Half ends just below the headline answer (minutes, arrival), leaving the
  // map most of the screen; the detail is a flick away.
  const half = Math.min(Math.max((bottomOf('.results .answer-top') || innerHeight * 0.5) + 18,
                                 peek + 100), innerHeight * 0.62);
  return { peek, half, full };
}

// Put the sheet at a rest (animated), or anywhere (instantly, mid-drag).
function showSheet(px, animate) {
  const sheet = document.querySelector('.plan');
  sheet.classList.toggle('dragging', !animate);
  sheet.style.transform = `translateY(${Math.round(sheet.offsetHeight - px)}px)`;
  SHEET_SHOWN = px;
  document.documentElement.style.setProperty('--sheet-visible', Math.round(px) + 'px');
}

function setSheet(rest, animate = true) {
  const sheet = document.querySelector('.plan');
  if (!PHONE.matches) {                 // a laptop: the plan is a side panel
    sheet.style.transform = '';
    delete sheet.dataset.sheet;
    return;
  }
  SHEET = rest;
  sheet.dataset.sheet = rest;
  if (rest !== 'full') sheet.scrollTop = 0;
  showSheet(sheetRests()[rest], animate);
  // Once it settles, frame the route in the map left above it. A timer, not
  // transitionend: a sheet already at its rest never fires one.
  clearTimeout(REFIT);
  REFIT = setTimeout(refitAboveSheet, animate ? 460 : 0);
}
let REFIT = 0;

function refitAboveSheet() {
  if (!PHONE.matches || SHEET === 'full' || !hasRoutes()) return;
  const sel = PLAN.routes[SELECTED];
  if (sel) MAP.fitBounds(L.polyline(sel.geometry).getBounds(), mapPadding());
}

function wireSheet() {
  const sheet = document.querySelector('.plan');

  // A tap on the grabber opens the sheet a step, or closes it from full.
  sheet.addEventListener('click', e => {
    if (PHONE.matches && e.clientY - sheet.getBoundingClientRect().top < 22) {
      setSheet(SHEET === 'full' ? 'half' : SHEET === 'half' ? 'full' : 'half');
    }
  }, true);

  // Typing needs the whole sheet, and room above the keyboard.
  sheet.addEventListener('focusin', e => {
    if (e.target.matches('input, select') && SHEET !== 'full') setSheet('full');
  });
  addEventListener('resize', () => setSheet(SHEET, false));
  PHONE.addEventListener('change', () => { setSheet(SHEET, false); MAP.invalidateSize(); });
  setSheet(SHEET, false);
}

/* ---------- mascot buddy ---------- */

// A friendly, low-stakes gimmick: click the mascot, get one short tip about
// using Triffy. It never fetches anything and never blocks the map or plan
// sheet - it only opens/closes a speech bubble next to itself.
const MASCOT_TIPS = [
  "Tap the swap arrows between the two fields to flip your trip in one go.",
  "Kolkata runs on modelled traffic over real roads; London is driven only by real TfL cameras. The badge above the search box always says which.",
  "Your route's colour shows the traffic you're expected to meet when you actually reach that stretch, not the traffic there right now.",
  "In Kolkata, try 'Arrive by' instead of 'Leave at' — I'll work backwards from your deadline.",
  "Tap any alternative route card to see it drawn on the map instead.",
  "The percentage next to your ETA is how predictable that route has been, not how fast it is.",
  "Pick how you're travelling — car, bike, auto or taxi — and I'll weigh routes the way that traveller actually would.",
  "On the London side, the tip tells you what share of your route is informed by live cameras versus typical conditions."
];

function wireMascot() {
  const btn = document.getElementById('mascotBtn');
  const bubble = document.getElementById('mascotBubble');
  const text = document.getElementById('mascotText');
  const close = document.getElementById('mascotBubbleClose');
  if (!btn || !bubble || !text || !close) return;

  let lastIndex = -1;
  const pickTip = () => {
    if (MASCOT_TIPS.length === 1) return MASCOT_TIPS[0];
    let i = Math.floor(Math.random() * MASCOT_TIPS.length);
    if (i === lastIndex) i = (i + 1) % MASCOT_TIPS.length;
    lastIndex = i;
    return MASCOT_TIPS[i];
  };

  const openBubble = () => {
    text.textContent = pickTip();
    bubble.hidden = false;
  };
  const closeBubble = () => { bubble.hidden = true; };

  btn.onclick = () => { bubble.hidden ? openBubble() : closeBubble(); };
  close.onclick = (e) => { e.stopPropagation(); closeBubble(); };

  // A tap anywhere else on the page dismisses the tip, like any other popover.
  document.addEventListener('click', (e) => {
    if (bubble.hidden) return;
    if (e.target === btn || btn.contains(e.target) || bubble.contains(e.target)) return;
    closeBubble();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !bubble.hidden) closeBubble();
  });
}

/* ---------- controls ---------- */

function wire() {
  document.querySelectorAll('button[data-mode]').forEach(b => {
    b.onclick = async () => {
      if (MODE === b.dataset.mode) return;
      MODE = b.dataset.mode;
      document.getElementById('advisory').classList.add('hidden');
      document.getElementById('results').innerHTML =
        `<p class="hint">Loading ${MODE === 'live' ? 'London' : 'Kolkata'}…</p>`;
      try {
        await loadCity();
        await plan();
      } catch (err) {
        showError((MODE === 'live' ? 'London' : 'Kolkata') +
          ' is unavailable right now: ' + err.message);
      }
    };
  });

  document.getElementById('odSummary').onclick = () => {
    setCompact(false);
    document.getElementById('origin').focus();
  };

  document.getElementById('swap').onclick = () => {
    const o = document.getElementById('origin'), d = document.getElementById('dest');
    [o.value, d.value] = [d.value, o.value];
    plan();
  };

  const toggle = document.getElementById('drawerToggle');
  toggle.onclick = () => {
    const off = document.body.classList.toggle('drawer-off');
    toggle.setAttribute('aria-expanded', String(!off));
    toggle.setAttribute('aria-label', off ? 'Show the map key' : 'Hide the map key');
    // Let the slide finish, then re-frame the route in the space it freed.
    setTimeout(() => { MAP.invalidateSize(); if (PLAN) drawRoutes(); }, 260);
  };

  document.getElementById('go').onclick = plan;
  document.getElementById('persona').onchange = () => { hintPersona(); plan(); };
  ['origin', 'dest', 'depart', 'arriveBy'].forEach(id => {
    document.getElementById(id).addEventListener('keydown', e => {
      if (e.key === 'Enter') plan();
    });
  });

  wireMascot();
  wireSheet();
}

/* ---------- util ---------- */

async function post(path, body) {
  const r = await fetch(API + path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || 'request failed');
  return j;
}

boot();

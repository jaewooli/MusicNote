/* MusicNote — shared front-end core (analysis page + editor page). */
'use strict';
const $ = s => document.querySelector(s);
let LAST = null, CUR = null, ROLL = null, ACTIVE_STEM = null;
let synthURL = null, synthToken = 0, TEMPO_TOUCHED = false;

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const midiName = m => { m = Math.round(m); return NOTE_NAMES[((m % 12) + 12) % 12] + (Math.floor(m / 12) - 1); };
const midiFreq = m => +(440 * Math.pow(2, (m - 69) / 12)).toFixed(2);

function curTS() { const el = $('#tsSel'); const p = ((el && el.value) || '4/4').split('/').map(Number); return [p[0] || 4, p[1] || 4]; }
function curTempo() { const el = $('#scoreTempo'); return (el && +el.value) || (CUR && CUR.tempo) || 120; }

function confColor(c) {
  if (typeof c !== 'number') return null;
  if (c >= 0.7) return '#7ee0b8';
  if (c >= 0.5) return '#ffcf6e';
  return '#ff8a8a';
}
function confCell(c) {
  if (typeof c !== 'number') return '-';
  return `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;` +
         `background:${confColor(c)};margin-right:5px;vertical-align:1px"></span>${Math.round(c * 100)}`;
}

// ---- job persistence: a page reload must NOT re-run analysis -----------------
const MN = {
  KEY: 'mn_job', TTL: 3.4 * 3600 * 1000,
  storeJob(id) {
    try { localStorage.setItem(this.KEY, JSON.stringify({ id, t: Date.now() })); } catch (_) {}
    try { history.replaceState(null, '', '#job=' + id); } catch (_) {}
  },
  storedJobId() {
    const h = (location.hash.match(/job=([0-9a-fA-F]{8,})/) || [])[1];
    if (h) return h;
    try {
      const j = JSON.parse(localStorage.getItem(this.KEY) || 'null');
      if (j && j.id && Date.now() - j.t < this.TTL) return j.id;
    } catch (_) {}
    return null;
  },
  clearJob() {
    try { localStorage.removeItem(this.KEY); } catch (_) {}
    try { history.replaceState(null, '', location.pathname); } catch (_) {}
  },
};

// ---- progress / status (guarded: editor page has neither) -------------------
function setStatus(html, cls) {
  const el = $('#status'); if (!el) return;
  el.innerHTML = html; el.className = 'status ' + (cls || '');
}
function showProgress(frac, msg, indet) {
  const p = $('#progress'); if (!p) return;
  p.style.display = 'block';
  $('#pmsg').textContent = msg;
  const pct = Math.max(0, Math.min(100, Math.round((frac || 0) * 100)));
  $('#ppct').textContent = indet ? '' : pct + '%';
  $('#pfill').style.width = (indet ? 35 : pct) + '%';
  $('#progress .bar').classList.toggle('indet', !!indet);
}
function hideProgress() {
  const p = $('#progress'); if (!p) return;
  p.style.display = 'none';
  $('#progress .bar').classList.remove('indet');
}

function pollJob(jobId) {
  return new Promise((resolve, reject) => {
    let stalls = 0;
    const tick = async () => {
      let p, r;
      try {
        r = await fetch('api/progress/' + jobId);
        p = await r.json();
      } catch (e) {
        if (++stalls > 6) return reject(new Error('서버 응답 없음'));
        return setTimeout(tick, 800);
      }
      if (!r.ok) return reject(new Error(p.error || p.detail || ('HTTP ' + r.status)));
      stalls = 0;
      if (p.status === 'running') {
        const pre = (p.steps > 1) ? `(${p.step}/${p.steps}) ` : '';
        showProgress(p.pct, pre + (p.message || '처리 중…'), false);
        setTimeout(tick, 350);
      } else if (p.status === 'done') {
        showProgress(1, '완료', false);
        setTimeout(hideProgress, 700);
        LAST = p.result;
        showResult(p.result);
        setStatus(p.result.warning ? ('⚠ ' + p.result.warning) : '완료 ✓',
          p.result.warning ? 'warn' : '');
        resolve(p.result);
      } else {
        hideProgress();
        reject(new Error(p.error || '분석 실패'));
      }
    };
    tick();
  });
}

// ---- results dispatch ------------------------------------------------------
function _render(d, fromRefine) { (window.render || renderCommon)(d, fromRefine); }

function showResult(r) {
  const res = $('#results'); if (res) res.style.display = 'block';
  let shown;
  if (Array.isArray(r.stems) && r.stems.length) {
    const sc = $('#stemCard'); if (sc) sc.style.display = '';
    renderStems(r);
    if (r.active_stem) { ACTIVE_STEM = r.active_stem; shown = stemView(r, ACTIVE_STEM); }
    else { ACTIVE_STEM = null; shown = r; }
  } else {
    const sc = $('#stemCard'); if (sc) sc.style.display = 'none';
    ACTIVE_STEM = null;
    shown = r;
  }
  _render(shown, false);
  Editor.load(shown);
}

function stemFor(r, id) { return (r.stems || []).find(s => s.id === id); }

function stemView(r, id) {
  const s = stemFor(r, id) || {};
  return {
    engine: s.engine, mode: (s.engine === 'basic-pitch' || s.engine === 'cqt-fallback')
      ? 'polyphonic' : 'melody',
    duration: r.duration, tempo: s.tempo || r.tempo || 0, note_count: (s.notes || []).length,
    notes: s.notes || [], contour: s.contour || [],
    sensitivity: typeof s.sensitivity === 'number' ? s.sensitivity : 0.5,
    quantized: !!s.quantized, beat_count: s.beat_count,
    key: s.key || null, low_conf: s.low_conf || 0,
    beats: r.beats || [], instrument: s.instrument, midi_url: s.midi_url,
    musicxml_url: r.musicxml_url || null, edited: !!r.edited,
    audio_url: r.audio_url,
    filename: (r.filename || '') + ' — ' + (s.label || id),
    job_id: r.job_id, _stem: id, warning: s.warning,
  };
}

const STEM_COLORS = ['#6ea8fe', '#7ee0b8', '#ffcf6e', '#ff8a8a', '#c99bff', '#7fd4ff', '#f2a1c8'];
function stemColor(r, id) {
  const i = (r.stems || []).findIndex(s => s.id === id);
  return i >= 0 ? STEM_COLORS[i % STEM_COLORS.length] : '#6ea8fe';
}

function renderStems(r) {
  const list = $('#stemList'); if (!list) return;
  const dur = r.duration || 1;
  const rows = r.stems.map((s, i) => {
    const segs = (s.spans || []).map(([a, b]) =>
      `<div class="seg" style="left:${a / dur * 100}%;width:${Math.max(0.4, (b - a) / dur * 100)}%;` +
      `background:${STEM_COLORS[i % STEM_COLORS.length]}"></div>`).join('');
    const pct = Math.round((s.presence || 0) * 100);
    const cnt = s.pitched ? `${(s.notes || []).length}음` : '음정 없음';
    const btn = (s.pitched && (s.notes || []).length)
      ? `<button class="ghost" data-stem="${s.id}">이 선율 보기</button>` : '';
    const mid = s.midi_url ? `<a class="ghost" href="${s.midi_url}" download>⬇ MIDI</a>` : '';
    const au = s.audio_url ? `<audio controls preload="none" playsinline src="${s.audio_url}"></audio>` : '';
    return `<div class="stem" data-id="${s.id}">
      <div class="top"><span class="nm"><span class="dot" style="background:${STEM_COLORS[i % STEM_COLORS.length]}"></span>${s.label}</span>
        <span class="sub">${cnt} · 곡의 ${pct}% 구간</span></div>
      <div class="timeline">${segs}</div>
      ${au}
      <div class="acts">${btn}${mid}</div>
    </div>`;
  }).join('');
  const allRow = (r.mode === 'polyphonic' || r.mode === 'mt3')
    ? `<div class="stem" data-id="__all__"><div class="top">
        <span class="nm">＝ 전체 합침 (모든 악기)</span>
        <span class="sub">${(r.notes || []).length}음</span></div>
      <div class="acts"><button class="ghost" data-stem="__all__">합친 채보 보기</button>
        ${r.midi_url ? `<a class="ghost" href="${r.midi_url}" download>⬇ MIDI</a>` : ''}</div></div>`
    : '';
  list.innerHTML = allRow + rows;
  list.querySelectorAll('button[data-stem]').forEach(b =>
    b.addEventListener('click', () => selectStem(b.dataset.stem)));
  markStem();
}

function markStem() {
  const list = $('#stemList'); if (!list) return;
  list.querySelectorAll('.stem').forEach(el =>
    el.classList.toggle('sel', el.dataset.id === (ACTIVE_STEM || '__all__')));
}

function selectStem(id) {
  if (!LAST || !LAST.stems) return;
  if (id === '__all__') {
    ACTIVE_STEM = null; markStem();
    _render(LAST, false); Editor.load(LAST);
    setStatus('전체 합침 채보 보기', '');
    return;
  }
  ACTIVE_STEM = id; markStem();
  const shown = stemView(LAST, id);
  _render(shown, false); Editor.load(shown);
  setStatus('스템 “' + (stemFor(LAST, id) || {}).label + '” 보기', '');
}

// ---- renderCommon: the part shared by both pages -------------------------
function renderCommon(d, fromRefine) {
  CUR = d;
  const nc = $('#ncount'); if (nc) nc.textContent = d.note_count;

  const tb = $('#tbody');
  if (tb) {
    const tagged = d.notes.some(n => n.inst);
    const hasConf = d.notes.some(n => typeof n.conf === 'number');
    const th1 = $('#instTh'), th2 = $('#confTh');
    if (th1) th1.hidden = !tagged;
    if (th2) th2.hidden = !hasConf;
    tb.innerHTML = d.notes.map((n, i) =>
      `<tr${n.conf < 0.5 ? ' style="background:rgba(255,138,138,.09)"' : ''}><td>${i + 1}</td>` +
      `${tagged ? `<td>${n.inst || '-'}</td>` : ''}<td>${n.name}</td><td>${n.pitch}</td><td>${n.freq}</td>` +
      `<td>${n.start.toFixed(3)}</td><td>${(n.end - n.start).toFixed(3)}</td><td>${n.velocity}</td>` +
      `${hasConf ? `<td>${confCell(n.conf)}</td>` : ''}</tr>`).join('');
  }

  const md = $('#midi');
  if (md) { md.style.display = d.midi_url ? '' : 'none'; if (d.midi_url) md.href = d.midi_url + '?v=' + Date.now(); }
  const mx = $('#mxmlDl');
  if (mx) {
    if (d.musicxml_url) { mx.href = d.musicxml_url + '?v=' + Date.now(); mx.style.display = ''; }
    else if (!fromRefine) mx.style.display = 'none';
  }
  const ks = $('#keySel');
  if (ks && ks.options[0]) ks.options[0].textContent = d.key ? `자동 (${d.key})` : '자동 (C)';
  const st = $('#scoreTempo');
  if (st && !fromRefine && !TEMPO_TOUCHED && +d.tempo) st.value = Math.round(d.tempo);

  const oe = $('#openEditor');
  if (oe && d.job_id) {
    const sp = ACTIVE_STEM || d._stem || '';
    oe.href = 'editor.html?job=' + d.job_id + (sp ? '&stem=' + encodeURIComponent(sp) : '');
    oe.style.display = '';
  }
  const ofs = $('#openFullScore');
  if (ofs && d.job_id) { ofs.href = 'score.html?job=' + d.job_id; ofs.style.display = ''; }

  const sw = $('#synthWrap');
  if (sw) sw.style.display = (d.notes && d.notes.length) ? '' : 'none';
  Player.setNotes(d.notes || [], d.duration || 0);   // instant — no re-render
  bindRoll();
  drawRoll(d);
  Score.render(d);
}

// ---- piano roll ----------------------------------------------------------
function drawRoll(d) {
  const cv = $('#roll'); if (!cv) return;
  const notes = d.notes;
  const ctx = cv.getContext('2d');
  const dur = Math.max(d.duration, ...notes.map(n => n.end), 1);
  let lo = 127, hi = 0;
  notes.forEach(n => { lo = Math.min(lo, n.pitch); hi = Math.max(hi, n.pitch); });
  if (hi < lo) { lo = 48; hi = 72; }
  lo -= 2; hi += 2;

  const pxPerSec = Math.max(40, Math.min(160, 1000 / dur));
  const W = Math.max(1000, Math.ceil(dur * pxPerSec));
  const rows = hi - lo + 1;
  const rh = Math.max(6, Math.min(16, Math.floor(420 / rows)));
  const H = rows * rh;
  cv.width = W; cv.height = H;
  ROLL = { pxPerSec, lo, hi, rh, W, H, dur };
  const ri = $('#rollInner'); if (ri) { ri.style.width = W + 'px'; ri.style.height = H + 'px'; }

  ctx.fillStyle = '#0b0d11'; ctx.fillRect(0, 0, W, H);
  for (let p = lo; p <= hi; p++) {
    const y = (hi - p) * rh;
    const isC = p % 12 === 0;
    ctx.fillStyle = ([1, 3, 6, 8, 10].includes(p % 12)) ? '#12151b' : '#0f1218';
    ctx.fillRect(0, y, W, rh);
    if (isC) {
      ctx.fillStyle = '#2b303b'; ctx.fillRect(0, y, W, 1);
      ctx.fillStyle = '#5b6472'; ctx.font = '10px system-ui';
      ctx.fillText('C' + (p / 12 - 1), 3, y + rh - 2);
    }
  }
  for (let s = 0; s <= dur; s++) {
    const x = s * pxPerSec;
    ctx.fillStyle = s % 5 === 0 ? '#2b303b' : '#171b22';
    ctx.fillRect(x, 0, 1, H);
  }

  const tagged = notes.some(n => n.stem);
  const editing = Editor.on && Editor.cur === d;
  notes.forEach(n => {
    const x = n.start * pxPerSec, w = Math.max(2, (n.end - n.start) * pxPerSec);
    const y = (hi - Math.round(n.pitch)) * rh;
    const a = 0.4 + 0.6 * (n.velocity / 127);
    const col = tagged ? (LAST ? stemColor(LAST, n.stem) : '#6ea8fe') : '#6ea8fe';
    ctx.globalAlpha = a; ctx.fillStyle = col;
    ctx.fillRect(x, y + 1, w, rh - 2);
    ctx.globalAlpha = 1;
    if (editing && Editor.sel.has(n)) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, y + 1.5, Math.max(2, w - 1), rh - 3);
      ctx.fillStyle = '#fff'; ctx.fillRect(x + w - 3, y + 1, 3, rh - 2);
      ctx.lineWidth = 1;
    } else if (typeof n.conf === 'number' && n.conf < 0.7) {
      ctx.strokeStyle = confColor(n.conf); ctx.lineWidth = n.conf < 0.5 ? 2 : 1;
      ctx.setLineDash(n.conf < 0.5 ? [3, 2] : []);
      ctx.strokeRect(x + 1, y + 1.5, Math.max(2, w - 1), rh - 3);
      ctx.setLineDash([]); ctx.lineWidth = 1;
    } else {
      ctx.strokeStyle = 'rgba(255,255,255,.25)'; ctx.strokeRect(x + .5, y + 1.5, w, rh - 3);
    }
  });
  if (tagged && LAST && LAST.stems) {
    let lx = 6;
    ctx.font = '10px system-ui';
    LAST.stems.filter(s => (s.notes || []).length).forEach((s) => {
      ctx.fillStyle = stemColor(LAST, s.id);
      ctx.fillRect(lx, 4, 9, 9);
      ctx.fillStyle = '#9aa3b2';
      const t = s.label.slice(0, 10);
      ctx.fillText(t, lx + 12, 12);
      lx += 12 + ctx.measureText(t).width + 12;
    });
  }
  if (d.contour && d.contour.length) {
    ctx.beginPath(); ctx.strokeStyle = 'rgba(255,207,110,.9)'; ctx.lineWidth = 1.5;
    d.contour.forEach((c, i) => {
      const x = c.t * pxPerSec, y = (hi - c.midi) * rh + rh / 2;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
  Playhead.move(Player.now());
}

// ---- live Web-Audio player: LOOK-AHEAD scheduler (no offline render) --------
// Switching notes is instant. Notes are scheduled ~0.3 s ahead in a 40 ms poll
// so envelopes are always in the future of a running clock (fixes "one blip
// then silence" that happened when everything was queued before ctx.resume()).
const Player = {
  ctx: null, notes: [], dur: 0, pos: 0, playing: false,
  _master: null, _startCtx: 0, _startPos: 0, _next: 0, _timer: 0, _raf: 0, _osc: [],
  _ac() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    return this.ctx;
  },
  setNotes(notes, dur) {
    this.notes = (notes || []).slice().sort((a, b) => a.start - b.start);
    this.dur = Math.max(dur || 0, ...this.notes.map(n => n.end), 0.1);
    if (this.pos > this.dur) this.pos = 0;
    if (this.playing) { const p = this.now(); this._halt(); this.pos = p; this._begin(); }
    else transportFrame(this.pos, this.dur);
  },
  now() {
    return this.playing
      ? Math.min(this.dur, this._startPos + (this.ctx.currentTime - this._startCtx))
      : this.pos;
  },
  _halt() {
    clearInterval(this._timer); cancelAnimationFrame(this._raf);
    this._timer = this._raf = 0;
    this._osc.forEach(o => { try { o.stop(); } catch (_) {} });
    this._osc = [];
    if (this._master) { try { this._master.disconnect(); } catch (_) {} this._master = null; }
  },
  _voice(n) {
    const ac = this.ctx;
    const when = Math.max(this._startCtx + (n.start - this._startPos), ac.currentTime + 0.004);
    const end = this._startCtx + (n.end - this._startPos);
    if (end <= ac.currentTime + 0.01) return;
    const dur = Math.max(0.03, end - when);
    const vpk = 0.045 + 0.5 * Math.pow((n.velocity || 90) / 127, 1.35);   // peak gain
    // per-note amplitude envelope from the backend (10-pt, own-peak-normalised),
    // so struck-and-decaying / swelling notes actually sound that way
    const e = (Array.isArray(n.env) && n.env.length >= 2)
      ? n.env : [124, 118, 100, 82, 66, 52, 40, 30, 20, 10];
    const curve = new Float32Array(e.length);
    for (let k = 0; k < e.length; k++) curve[k] = Math.max(1e-4, (e[k] / 127) * vpk);
    const g = ac.createGain();
    const curveEnd = when + dur, tail = curveEnd + 0.05;
    g.gain.setValueAtTime(curve[0], when);
    try { g.gain.setValueCurveAtTime(curve, when, dur); }
    catch (_) {
      for (let k = 1; k < curve.length; k++)
        g.gain.linearRampToValueAtTime(curve[k], when + dur * (k / (curve.length - 1)));
    }
    g.gain.linearRampToValueAtTime(1e-4, tail);                  // clean tail
    g.connect(this._master);
    [[1, 1.0], [2, 0.35], [3, 0.16]].forEach(([m, a]) => {
      const o = ac.createOscillator(), pg = ac.createGain();
      o.type = 'sine'; o.frequency.value = n.freq * m; pg.gain.value = a;
      o.connect(pg).connect(g);
      o.start(when); o.stop(tail + 0.02);
      this._osc.push(o);
    });
  },
  _pump() {
    if (!this.playing) return;
    const songT = this._startPos + (this.ctx.currentTime - this._startCtx);
    while (this._next < this.notes.length && this.notes[this._next].start < songT + 0.3) {
      const n = this.notes[this._next++];
      if (n.end > songT - 0.02) this._voice(n);
    }
    if (songT >= this.dur) this.stop();
  },
  _begin() {
    const ac = this._ac();
    const go = () => {
      this._startPos = this.pos >= this.dur ? 0 : this.pos;
      this.pos = this._startPos;
      this._startCtx = ac.currentTime + 0.08;
      this._master = ac.createGain(); this._master.gain.value = 0.5;
      this._master.connect(ac.destination);
      this._next = this.notes.findIndex(n => n.start >= this._startPos - 0.02);
      if (this._next < 0) this._next = this.notes.length;
      this.playing = true;
      transportState(true);
      this._timer = setInterval(() => this._pump(), 40);
      const frame = () => { if (!this.playing) return; transportFrame(this.now(), this.dur); this._raf = requestAnimationFrame(frame); };
      this._raf = requestAnimationFrame(frame);
      this._pump();
    };
    if (ac.state === 'suspended') ac.resume().then(go, go); else go();
  },
  play() { if (!this.playing && this.notes.length) this._begin(); },
  pause() { if (this.playing) { this.pos = this.now(); this._halt(); this.playing = false; transportState(false); transportFrame(this.pos, this.dur); } },
  toggle() { this.playing ? this.pause() : this.play(); },
  stop() { this._halt(); this.playing = false; this.pos = 0; transportState(false); transportFrame(0, this.dur); },
  seek(t) {
    const was = this.playing;
    this._halt(); this.playing = false;
    this.pos = Math.max(0, Math.min(this.dur, t));
    if (was) this._begin(); else { transportState(false); transportFrame(this.pos, this.dur); }
  },
};

const _fmtT = s => { s = Math.max(0, s | 0); return (s / 60 | 0) + ':' + String(s % 60).padStart(2, '0'); };
let _seekDragging = false;
function transportState(playing) {
  const b = $('#ppBtn'); if (b) b.textContent = playing ? '⏸' : '▶';
}
function transportFrame(t, dur) {
  const tp = $('#tpos'); if (tp) tp.textContent = _fmtT(t) + ' / ' + _fmtT(dur);
  const sk = $('#seek'); if (sk && !_seekDragging) sk.value = dur ? Math.round(t / dur * 1000) : 0;
  Playhead.move(t);
}
(function wireTransport() {
  const pp = $('#ppBtn'), sk = $('#seek');
  if (pp) pp.addEventListener('click', () => Player.toggle());
  if (sk) {
    const apply = () => Player.seek((+sk.value) / 1000 * Player.dur);
    sk.addEventListener('pointerdown', () => { _seekDragging = true; });
    sk.addEventListener('input', () => {
      const t = (+sk.value) / 1000 * Player.dur;
      const tp = $('#tpos'); if (tp) tp.textContent = _fmtT(t) + ' / ' + _fmtT(Player.dur);
      Playhead.move(t); // scrub preview must move both roll and score
    });
    const finishSeek = () => { _seekDragging = false; apply(); };
    sk.addEventListener('change', finishSeek);
    sk.addEventListener('pointerup', finishSeek);
    sk.addEventListener('pointercancel', finishSeek);
    sk.addEventListener('keydown', e => { if (e.key.startsWith('Arrow')) finishSeek(); });
  }
})();

// ---- playhead on the roll + the score --------------------------------------
const Playhead = {
  move(t) {
    const rh = $('#rollHead');
    if (rh && ROLL) {
      const x = t * ROLL.pxPerSec;
      rh.hidden = !Player.playing && t <= 0;
      rh.style.transform = 'translateX(' + x + 'px)';
      if (Player.playing) {
        const wrap = $('#pianoWrap');
        const want = x - wrap.clientWidth * 0.35;
        if (want > wrap.scrollLeft || x > wrap.scrollLeft + wrap.clientWidth - 40)
          wrap.scrollLeft = Math.max(0, want);
      }
    }
    const sh = $('#scoreHead'), L = Score._layout;
    if (sh && L && L.length) {
      const m = L.find(z => t >= z.tStart && t < z.tEnd) || (t <= 0 ? L[0] : null);
      if (!m) { sh.hidden = true; return; }
      const x = m.x + (t - m.tStart) / (m.tEnd - m.tStart) * m.w;
      sh.hidden = !Player.playing && t <= 0;
      sh.style.left = x + 'px';
      sh.style.top = m.y + 'px';
      sh.style.height = m.h + 'px';
      if (Player.playing) {
        const box = $('#score') || $('#fullScore');
        if (box && (x < box.scrollLeft || x > box.scrollLeft + box.clientWidth - 30))
          box.scrollLeft = Math.max(0, x - box.clientWidth * 0.3);
        // follow vertically when the score is a tall multi-system page
        if ($('#fullScore') && !$('#score')) {
          const py = sh.getBoundingClientRect().top;
          if (py < 90 || py > window.innerHeight - 90)
            window.scrollBy({ top: py - window.innerHeight * 0.35, behavior: 'instant' });
        }
      }
    }
  },
};

// ===================== Notation (VexFlow) ==============================
function topContour(notes) {
  if (!notes || !notes.length) return [];
  const end = Math.max(...notes.map(n => n.end));
  const out = [];
  for (let t = 0; t < end; t += 0.05) {
    let top = null;
    for (const n of notes) if (n.start <= t && t < n.end && (top === null || n.pitch > top)) top = Math.round(n.pitch);
    if (top !== null) out.push({ t: +t.toFixed(3), midi: top, freq: midiFreq(top) });
  }
  return out;
}

const DUR_UNITS = [[16, 'w', 0], [12, 'h', 1], [8, 'h', 0], [6, 'q', 1], [4, 'q', 0], [3, '8', 1], [2, '8', 0], [1, '16', 0]];
function decompose(units) {
  const out = []; let rem = Math.round(units), guard = 0;
  while (rem > 0 && guard++ < 64) {
    const hit = DUR_UNITS.find(d => d[0] <= rem);
    if (!hit) break;
    out.push([hit[1], hit[2]]); rem -= hit[0];
  }
  return out.length ? out : [['16', 0]];
}

function buildMeasures(notes, tempo, num, den) {
  const secPerQ = 60 / (tempo || 120);
  const sec16 = secPerQ / 4;
  const upm = Math.max(1, num * (16 / den));
  const ns = (notes || []).filter(n => n.end > n.start)
    .slice().sort((a, b) => a.start - b.start || b.pitch - a.pitch);
  const totalSec = ns.length ? Math.max(...ns.map(n => n.end)) : secPerQ * num;
  let totalUnits = Math.max(upm, Math.round(totalSec / sec16) + 1);
  totalUnits = Math.min(totalUnits, upm * 400);
  const grid = new Array(totalUnits).fill(null);
  ns.forEach(n => {
    const g0 = Math.max(0, Math.round(n.start / sec16));
    const g1 = Math.max(g0 + 1, Math.round(n.end / sec16));
    const p = Math.round(n.pitch);
    for (let g = g0; g < Math.min(g1, totalUnits); g++)
      if (grid[g] === null || p > grid[g]) grid[g] = p;
  });
  const nMeas = Math.ceil(totalUnits / upm);
  const measures = [];
  for (let m = 0; m < nMeas; m++) {
    const segs = [], start = m * upm, end = start + upm;
    let i = start;
    while (i < end) {
      const p = i < totalUnits ? grid[i] : null;
      let j = i;
      while (j < end && (j < totalUnits ? grid[j] : null) === p) j++;
      segs.push([p, j - i]); i = j;
    }
    measures.push(segs);
  }
  const voiced = grid.filter(p => p != null).sort((a, b) => a - b);
  const clef = voiced.length && voiced[voiced.length >> 1] < 57 ? 'bass' : 'treble';
  return { measures, clef, upm };
}

const SHARP_LET = ['c', 'c', 'd', 'd', 'e', 'f', 'f', 'g', 'g', 'a', 'a', 'b'];
const SHARP_AL = ['', '#', '', '#', '', '', '#', '', '#', '', '#', ''];
const FLAT_LET = ['c', 'd', 'd', 'e', 'e', 'f', 'g', 'g', 'a', 'a', 'b', 'b'];
const FLAT_AL = ['', 'b', '', 'b', '', '', 'b', '', 'b', '', 'b', ''];
const KEYSIG = {
  C: { t: null, s: new Set() }, G: { t: '#', s: new Set([6]) },
  D: { t: '#', s: new Set([6, 1]) }, A: { t: '#', s: new Set([6, 1, 8]) },
  E: { t: '#', s: new Set([6, 1, 8, 3]) }, B: { t: '#', s: new Set([6, 1, 8, 3, 10]) },
  F: { t: 'b', s: new Set([10]) }, Bb: { t: 'b', s: new Set([10, 3]) },
  Eb: { t: 'b', s: new Set([10, 3, 8]) }, Ab: { t: 'b', s: new Set([10, 3, 8, 1]) },
};
const MINOR_TO_MAJOR = { A: 'C', E: 'G', B: 'D', 'F#': 'A', 'C#': 'E', D: 'F', G: 'Bb', C: 'Eb', F: 'Ab' };
function effectiveKey(d) {
  const el = $('#keySel');
  const sel = (el && el.value) || 'auto';
  if (sel !== 'auto') return sel;
  const m = ((d && d.key) || '').match(/^([A-G][#b]?)\s+(major|minor)$/i);
  if (!m) return 'C';
  let name = m[1];
  if (m[2].toLowerCase() === 'minor') name = MINOR_TO_MAJOR[name] || 'C';
  return KEYSIG[name] ? name : 'C';
}
function spell(midi, keyName) {
  const key = KEYSIG[keyName] || KEYSIG.C;
  const pc = ((midi % 12) + 12) % 12, oct = Math.floor(midi / 12) - 1;
  const flat = key.t === 'b';
  const letter = (flat ? FLAT_LET : SHARP_LET)[pc], alt = (flat ? FLAT_AL : SHARP_AL)[pc];
  let acc = null;
  if (alt) acc = key.s.has(pc) ? null : (flat ? 'b' : '#');
  else if (key.t === '#' && key.s.has((pc + 1) % 12)) acc = 'n';
  else if (key.t === 'b' && key.s.has((pc + 11) % 12)) acc = 'n';
  return { vexKey: letter + alt + '/' + oct, acc };
}

const Score = {
  timer: null, _layout: [], _hit: [], _d: null, _staff: null, _docSeq: 0,
  render(d) {
    clearTimeout(this.timer);
    this._d = d;
    // The result page is read-only, so it can render the server-built ScoreDoc
    // directly. That keeps its notation (chords, tuplets, ties and durations)
    // identical to the MusicXML export. The editor retains the legacy renderer
    // for now because it supplies the hit map used for direct note manipulation.
    if (!Editor.on && d && d.job_id) {
      const seq = ++this._docSeq;
      const box = $('#scoreSvg') || $('#score');
      if (!box) return;
      const params = {};
      if (d._stem) params.stem = d._stem;
      const ts = curTS();
      params.num = ts[0]; params.den = ts[1]; params.tempo = curTempo();
      const key = $('#keySel');
      if (key && key.value !== 'auto') {
        const fifths = { C: 0, G: 1, D: 2, A: 3, E: 4, B: 5, F: -1, Bb: -2, Eb: -3, Ab: -4 };
        params.fifths = fifths[key.value] ?? 0;
      }
      fetchScoreDoc(d.job_id, params).then(doc => {
        if (seq !== this._docSeq || this._d !== d) return;
        renderDoc(box, doc, { cap: 240 });
      }).catch(e => {
        if (seq !== this._docSeq || this._d !== d) return;
        box.innerHTML = '<div class="hint" style="padding:18px">악보 생성 실패: '
          + String(e.message || e) + '</div>';
      });
      return;
    }
    this.timer = setTimeout(() => this._go(d), 240);
  },
  _go(d) {
    const box = $('#scoreSvg') || $('#score');
    if (!box) return;
    this._d = d; this._layout = []; this._hit = [];
    if (!window.Vex) { box.innerHTML = '<div class="hint" style="padding:18px">악보 렌더러(vexflow) 로드 실패</div>'; return; }
    if (!d || !d.notes || !d.notes.length) { box.innerHTML = '<div class="hint" style="padding:18px">표시할 음표가 없습니다.</div>'; return; }
    try { this._draw(box, d); }
    catch (e) { box.innerHTML = '<div class="hint" style="padding:18px">악보 생성 실패: ' + e.message + '</div>'; }
    scoreBind();
  },
  _draw(box, d) {
    const VF = Vex.Flow;
    box.innerHTML = '';
    const tempo = curTempo();
    const sec16 = (60 / (tempo || 120)) / 4;
    const ts = curTS(), num = ts[0], den = ts[1];
    const keyName = effectiveKey(d);
    const showKey = keyName !== 'C';
    const lowSpans = d.notes.filter(n => n.conf < 0.5).map(n => [n.start, n.end]);
    const isLow = (t0, t1) => lowSpans.some(sp => t0 < sp[1] && t1 > sp[0]);
    const selT = [...Editor.sel].map(n => [n.start, n.end, Math.round(n.pitch)]);
    const isSel = (t0, t1, p) => selT.some(s => s[2] === p && t0 < s[1] && t1 > s[0]);
    const built = buildMeasures(d.notes, tempo, num, den);
    const CAP = 240;
    const ms = built.measures.slice(0, CAP);
    const W = Math.max(340, Math.min(($('#score') || box).clientWidth || 900, 1100));
    const per = Math.max(1, Math.min(4, Math.floor(W / 250)));
    const sw = Math.floor((W - 16) / per);
    const lines = Math.ceil(ms.length / per);
    const H = lines * 132 + 24;
    const renderer = new VF.Renderer(box, VF.Renderer.Backends.SVG);
    renderer.resize(W, H);
    const ctx = renderer.getContext();
    ctx.setFont('Arial', 10);
    this._staff = { clef: built.clef, spacing: 10 };

    ms.forEach((segs, mi) => {
      const line = Math.floor(mi / per), col = mi % per;
      const x = 8 + col * sw, y = 12 + line * 132;
      const stave = new VF.Stave(x, y, sw);
      if (col === 0) { stave.addClef(built.clef); if (showKey) stave.addKeySignature(keyName); }
      if (mi === 0) stave.addTimeSignature(num + '/' + den);
      stave.setContext(ctx).draw();

      const notesX = col === 0 ? (showKey ? x + 110 : x + 68) : x + 14;
      const notesW = Math.max(20, x + sw - 10 - notesX);
      this._layout.push({ x: notesX, w: notesW, y, h: 118,
        tStart: mi * built.upm * sec16, tEnd: (mi + 1) * built.upm * sec16 });
      try { this._staff.spacing = stave.getSpacingBetweenLines() || 10; } catch (_) {}
      try { this._staff.topY = stave.getYForLine(0); } catch (_) { this._staff.topY = y + 20; }

      const vf = [], ties = [];
      let unit = mi * built.upm;
      segs.forEach(([pitch, len]) => {
        const t0 = unit * sec16, t1 = (unit + len) * sec16;
        unit += len;
        const comps = decompose(len);
        const low = pitch !== null && isLow(t0, t1);
        const sel = pitch !== null && isSel(t0, t1, pitch);
        comps.forEach(([code, dots], ci) => {
          let n;
          if (pitch === null) {
            n = new VF.StaveNote({ clef: built.clef, keys: [built.clef === 'bass' ? 'd/3' : 'b/4'], duration: code + 'r' });
          } else {
            const sp = spell(pitch, keyName);
            n = new VF.StaveNote({ clef: built.clef, keys: [sp.vexKey], duration: code });
            if (sp.acc) n.addModifier(new VF.Accidental(sp.acc), 0);
            if (sel) n.setStyle({ fillStyle: '#6ea8fe', strokeStyle: '#6ea8fe' });
            else if (low) n.setStyle({ fillStyle: '#c85f3b', strokeStyle: '#c85f3b' });
          }
          if (dots) VF.Dot.buildAndAttach([n], { all: true });
          n._srcPitch = pitch; n._t0 = t0; n._t1 = t1;
          vf.push(n);
          if (pitch !== null && comps.length > 1 && ci > 0) ties.push([vf[vf.length - 2], n]);
        });
      });

      const voice = new VF.Voice({ num_beats: num, beat_value: den }).setMode(VF.Voice.Mode.SOFT);
      voice.addTickables(vf);
      const pad = col === 0 ? (showKey ? 116 : 74) : 22;
      new VF.Formatter().joinVoices([voice]).format([voice], Math.max(40, sw - pad));
      voice.draw(ctx, stave);
      try { VF.Beam.generateBeams(vf.filter(n => !n.isRest())).forEach(b => b.setContext(ctx).draw()); } catch (_) {}
      ties.forEach(pair => new VF.StaveTie({ first_note: pair[0], last_note: pair[1], first_indices: [0], last_indices: [0] }).setContext(ctx).draw());

      vf.forEach(n => {
        if (n._srcPitch === null) return;
        let bb; try { bb = n.getBoundingBox(); } catch (_) { return; }
        if (bb) this._hit.push({ x: bb.x, y: bb.y, w: bb.w || 12, h: bb.h || 18,
          pitch: n._srcPitch, t0: n._t0, t1: n._t1 });
      });
    });
    if (built.measures.length > CAP)
      box.insertAdjacentHTML('beforeend', '<div class="hint" style="padding:6px">앞 ' + CAP + '마디만 표시</div>');
  },
  // score coord -> approximate diatonic MIDI pitch, snapped to the current key
  yToPitch(sy) {
    const st = this._staff; if (!st || st.topY == null) return 60;
    const WHITE = [0, 2, 4, 5, 7, 9, 11];
    const ref = st.clef === 'bass' ? { midi: 57, wi: 5, oct: 3 } : { midi: 77, wi: 3, oct: 5 };
    const steps = Math.round((sy - st.topY) / (st.spacing / 2));   // + = down = lower
    let wi = ref.wi - steps;
    const octShift = Math.floor(wi / 7);
    wi = ((wi % 7) + 7) % 7;
    let midi = 12 * ((ref.oct + octShift) + 1) + WHITE[wi];
    const key = KEYSIG[effectiveKey(this._d)] || KEYSIG.C;
    const allowed = new Set([...Array(12).keys()].filter(pc => {
      const flat = key.t === 'b';
      return !((flat ? FLAT_AL : SHARP_AL)[pc]) || key.s.has(pc);
    }));
    if (!allowed.has(((midi % 12) + 12) % 12)) {
      for (const d of [1, -1, 2, -2]) if (allowed.has((((midi + d) % 12) + 12) % 12)) { midi += d; break; }
    }
    return Math.max(21, Math.min(108, midi));
  },
};

// ======================= ScoreDoc renderer ==============================
// Draws the ScoreDoc the BACKEND built (GET /api/score/<job>), so the screen
// and the MusicXML export are the same notation — no re-derivation here.
const VF_DUR = {
  breve: 'w', whole: 'w', half: 'h', quarter: 'q',
  eighth: '8', '16th': '16', '32nd': '32', '64th': '64',
};
const VF_ACC = { 1: '#', 2: '##', '-1': 'b', '-2': 'bb' };

function _vfKey(n) {
  const acc = n.alter > 0 ? '#'.repeat(n.alter) : n.alter < 0 ? 'b'.repeat(-n.alter) : '';
  return n.step.toLowerCase() + acc + '/' + n.octave;
}

function renderDoc(box, doc, opts) {
  opts = opts || {};
  Score._layout = [];
  if (!window.Vex) { box.innerHTML = '<p class="hint" style="padding:20px">악보 렌더러 로드 실패</p>'; return; }
  const VF = Vex.Flow;
  const parts = (doc.parts || []).filter(p => p.voices && p.voices.length);
  if (!parts.length) { box.innerHTML = '<p class="hint" style="padding:20px">표시할 음표가 없습니다.</p>'; return; }

  const nMeas = Math.min(opts.cap || 300,
    Math.max(...parts.map(p => Math.max(...p.voices.map(v => v.measures.length)))));
  const secPerQ = 60 / (doc.tempo || 120);
  const quartersPerMeasure = doc.time_sig[0] * 4 / doc.time_sig[1];
  const fallbackMeasureSec = quartersPerMeasure * secPerQ;
  const measureTime = mi => {
    for (const p of parts) for (const v of p.voices) {
      const m = v.measures[mi];
      if (m && Number.isFinite(m.start) && Number.isFinite(m.end) && m.end > m.start)
        return [m.start, m.end];
    }
    return [mi * fallbackMeasureSec, (mi + 1) * fallbackMeasureSec];
  };

  const W = Math.max(720, Math.min(box.clientWidth || 1000, 1600));
  const partH = opts.partH || 96, sysGap = 36, labelW = parts.length > 1 ? 104 : 8, margin = 16;
  const per = Math.max(1, Math.min(6, Math.floor((W - margin * 2 - labelW) / 210)));
  const sw = Math.floor((W - margin * 2 - labelW) / per);
  const systems = Math.ceil(nMeas / per);
  const H = systems * (parts.length * partH + sysGap) + 30;

  box.innerHTML = '';
  const renderer = new VF.Renderer(box, VF.Renderer.Backends.SVG);
  renderer.resize(W, H);
  const ctx = renderer.getContext();
  ctx.setFont('Arial', 9);
  const keyName = fifthsToKey(doc.key_fifths);

  for (let mi = 0; mi < nMeas; mi++) {
    const sys = Math.floor(mi / per), col = mi % per;
    const x = margin + labelW + col * sw;
    const sysY = 16 + sys * (parts.length * partH + sysGap);
    const staves = [];
    parts.forEach((p, pi) => {
      const y = sysY + pi * partH;
      const st = new VF.Stave(x, y, sw);
      if (col === 0) {
        st.addClef(p.clef || 'treble');
        if (keyName !== 'C') st.addKeySignature(keyName);
      }
      if (mi === 0) st.addTimeSignature(doc.time_sig[0] + '/' + doc.time_sig[1]);
      st.setContext(ctx).draw();
      staves.push(st);
      if (col === 0 && parts.length > 1 && p.name) {
        try { ctx.save(); ctx.setFont('Arial', 10);
          ctx.fillText(String(p.name).slice(0, 12), 6, y + partH * 0.42); ctx.restore(); } catch (_) {}
      }

      const voices = [], allTuplets = [], allTies = [];
      p.voices.forEach(v => {
        const meas = v.measures[mi];
        if (!meas || !meas.events.length) return;
        const vf = [];
        meas.events.forEach(e => {
          const code = VF_DUR[e.type] || 'q';
          let n;
          if (!e.notes || !e.notes.length) {
            n = new VF.StaveNote({ clef: p.clef || 'treble',
              keys: [(p.clef === 'bass') ? 'd/3' : 'b/4'], duration: code + 'r' });
          } else {
            const keys = e.notes.map(_vfKey);
            n = new VF.StaveNote({ clef: p.clef || 'treble', keys, duration: code });
            e.notes.forEach((nn, i) => {
              if (nn.alter && VF_ACC[nn.alter])
                n.addModifier(new VF.Accidental(VF_ACC[nn.alter]), i);
            });
            if (e.notes.some(nn => nn.conf !== undefined && nn.conf < 0.5))
              n.setStyle({ fillStyle: '#c85f3b', strokeStyle: '#c85f3b' });
          }
          if (e.dots) VF.Dot.buildAndAttach([n], { all: true });
          n._ev = e;
          vf.push(n);
        });
        if (!vf.length) return;
        const voice = new VF.Voice({ num_beats: doc.time_sig[0], beat_value: doc.time_sig[1] })
          .setMode(VF.Voice.Mode.SOFT);
        voice.addTickables(vf);
        voices.push(voice);
        // tuplet brackets
        let i = 0;
        while (i < vf.length) {
          if (!vf[i]._ev.tuplet) { i++; continue; }
          let j = i;
          while (j + 1 < vf.length && vf[j + 1]._ev.tuplet
                 && !vf[j]._ev.tuplet_stop) j++;
          const grp = vf.slice(i, j + 1);
          if (grp.length > 1) {
            const t = vf[i]._ev.tuplet;
            try { allTuplets.push(new VF.Tuplet(grp, { num_notes: t[0], notes_occupied: t[1] })); } catch (_) {}
          }
          i = j + 1;
        }
        // ties inside the measure
        for (let k = 0; k + 1 < vf.length; k++) {
          const a = vf[k]._ev, b = vf[k + 1]._ev;
          if (a.notes && a.notes.length && a.notes[0].tie_start
              && b.notes && b.notes.length && b.notes[0].tie_stop)
            allTies.push([vf[k], vf[k + 1]]);
        }
      });
      if (!voices.length) return;
      const pad = col === 0 ? (keyName !== 'C' ? 108 : 66) : 18;
      try {
        new VF.Formatter().joinVoices(voices).format(voices, Math.max(36, sw - pad));
      } catch (_) {
        voices.forEach(v => { try { new VF.Formatter().joinVoices([v]).format([v], Math.max(36, sw - pad)); } catch (_) {} });
      }
      voices.forEach(v => { try { v.draw(ctx, st); } catch (_) {} });
      voices.forEach(v => {
        try {
          VF.Beam.generateBeams(v.getTickables().filter(z => !z.isRest() && !z._ev.tuplet))
            .forEach(b => b.setContext(ctx).draw());
        } catch (_) {}
      });
      allTuplets.forEach(t => { try { t.setContext(ctx).draw(); } catch (_) {} });
      allTies.forEach(pr => { try {
        new VF.StaveTie({ first_note: pr[0], last_note: pr[1], first_indices: [0], last_indices: [0] })
          .setContext(ctx).draw(); } catch (_) {} });
    });

    const notesX = col === 0 ? (keyName !== 'C' ? x + 106 : x + 64) : x + 12;
    const [tStart, tEnd] = measureTime(mi);
    Score._layout.push({
      x: notesX, w: Math.max(20, x + sw - 8 - notesX),
      y: sysY - 6, h: parts.length * partH + 12,
      tStart, tEnd,
    });
    if (col === 0 && staves.length > 1) {
      try {
        new VF.StaveConnector(staves[0], staves[staves.length - 1])
          .setType(VF.StaveConnector.type.BRACKET).setContext(ctx).draw();
        new VF.StaveConnector(staves[0], staves[staves.length - 1])
          .setType(VF.StaveConnector.type.SINGLE_LEFT).setContext(ctx).draw();
      } catch (_) {}
    }
  }
}

function fifthsToKey(f) {
  return ({ 0: 'C', 1: 'G', 2: 'D', 3: 'A', 4: 'E', 5: 'B',
    '-1': 'F', '-2': 'Bb', '-3': 'Eb', '-4': 'Ab' })[f] || 'C';
}

async function fetchScoreDoc(jobId, params) {
  const q = new URLSearchParams(params || {});
  const r = await fetch(`api/score/${jobId}?` + q.toString());
  if (!r.ok) throw new Error((await r.json()).detail || ('HTTP ' + r.status));
  return r.json();
}

// --- full multi-instrument score (one staff per part, stacked into systems) --
function renderFullScore(box, parts, opts) {
  Score._layout = [];
  if (!window.Vex) { box.innerHTML = '<p class="hint" style="padding:20px">악보 렌더러 로드 실패</p>'; return; }
  const VF = Vex.Flow;
  const tempo = opts.tempo || 120, num = opts.num || 4, den = opts.den || 4;
  const keyName = opts.keyName && KEYSIG[opts.keyName] ? opts.keyName : 'C';
  const showKey = keyName !== 'C';
  const P = (parts || []).filter(p => p.notes && p.notes.length);
  if (!P.length) { box.innerHTML = '<p class="hint" style="padding:20px">표시할 음표가 없습니다.</p>'; return; }

  const built = P.map(p => buildMeasures(p.notes, tempo, num, den));
  const sec16 = (60 / (tempo || 120)) / 4;
  const upm0 = built[0].upm;
  const CAP = 300;
  const nMeas = Math.min(CAP, Math.max(...built.map(b => b.measures.length)));
  const cw = (($('#fullScore') || box).clientWidth) || 1000;
  const W = Math.max(720, Math.min(cw, 1600));
  const partH = 96, sysGap = 36, labelW = 104, margin = 16;
  const per = Math.max(1, Math.min(6, Math.floor((W - margin * 2 - labelW) / 200)));
  const sw = Math.floor((W - margin * 2 - labelW) / per);
  const systems = Math.ceil(nMeas / per);
  const H = systems * (P.length * partH + sysGap) + 30;

  box.innerHTML = '';
  const renderer = new VF.Renderer(box, VF.Renderer.Backends.SVG);
  renderer.resize(W, H);
  const ctx = renderer.getContext();
  ctx.setFont('Arial', 9);

  for (let mi = 0; mi < nMeas; mi++) {
    const sys = Math.floor(mi / per), col = mi % per;
    const x = margin + labelW + col * sw;
    const sysY = 16 + sys * (P.length * partH + sysGap);
    const staves = [];
    P.forEach((part, pi) => {
      const b = built[pi];
      const segs = b.measures[mi] || [[null, b.upm]];
      const y = sysY + pi * partH;
      const st = new VF.Stave(x, y, sw);
      if (col === 0) { st.addClef(b.clef); if (showKey) st.addKeySignature(keyName); }
      if (mi === 0) st.addTimeSignature(num + '/' + den);
      st.setContext(ctx).draw();
      staves.push(st);
      if (col === 0 && part.label) {
        try {
          ctx.save();
          ctx.setFont('Arial', 10);
          ctx.fillText(part.label.slice(0, 12), 6, y + partH * 0.42);
          ctx.restore();
        } catch (_) {}
      }

      const vf = [], ties = [];
      segs.forEach(([pitch, len]) => {
        decompose(len).forEach(([code, dots], ci, arr) => {
          let n;
          if (pitch === null) {
            n = new VF.StaveNote({ clef: b.clef, keys: [b.clef === 'bass' ? 'd/3' : 'b/4'], duration: code + 'r' });
          } else {
            const sp = spell(pitch, keyName);
            n = new VF.StaveNote({ clef: b.clef, keys: [sp.vexKey], duration: code });
            if (sp.acc) n.addModifier(new VF.Accidental(sp.acc), 0);
          }
          if (dots) VF.Dot.buildAndAttach([n], { all: true });
          vf.push(n);
          if (pitch !== null && arr.length > 1 && ci > 0) ties.push([vf[vf.length - 2], n]);
        });
      });
      const voice = new VF.Voice({ num_beats: num, beat_value: den }).setMode(VF.Voice.Mode.SOFT);
      voice.addTickables(vf);
      const pad = col === 0 ? (showKey ? 108 : 66) : 18;
      new VF.Formatter().joinVoices([voice]).format([voice], Math.max(36, sw - pad));
      voice.draw(ctx, st);
      try { VF.Beam.generateBeams(vf.filter(z => !z.isRest())).forEach(bm => bm.setContext(ctx).draw()); } catch (_) {}
      ties.forEach(pr => new VF.StaveTie({ first_note: pr[0], last_note: pr[1], first_indices: [0], last_indices: [0] }).setContext(ctx).draw());
    });
    {
      const notesX = col === 0 ? (showKey ? x + 106 : x + 64) : x + 12;
      Score._layout.push({
        x: notesX, w: Math.max(20, x + sw - 8 - notesX),
        y: sysY - 6, h: P.length * partH + 12,
        tStart: mi * upm0 * sec16, tEnd: (mi + 1) * upm0 * sec16,
      });
    }
    if (col === 0 && staves.length > 1) {
      try {
        new VF.StaveConnector(staves[0], staves[staves.length - 1]).setType(VF.StaveConnector.type.BRACKET).setContext(ctx).draw();
        new VF.StaveConnector(staves[0], staves[staves.length - 1]).setType(VF.StaveConnector.type.SINGLE_LEFT).setContext(ctx).draw();
      } catch (_) {}
    }
  }
  if (Math.max(...built.map(b => b.measures.length)) > CAP)
    box.insertAdjacentHTML('beforeend', '<p class="hint" style="padding:6px">앞 ' + CAP + '마디만 표시</p>');
}

// ===================== interactive piano-roll editor ===================
const Editor = {
  on: false, cur: null, sel: new Set(), orig: '[]', undo: [], redo: [], drag: null, _pt: null,
  onChange: null,
  load(d) {
    this.cur = d; this.sel.clear(); this.undo = []; this.redo = [];
    this.orig = JSON.stringify((d && d.notes) || []);
    const bar = $('#editBar'); if (bar) bar.style.display = this.on ? 'flex' : 'none';
  },
  grid() {
    const spb = 60 / ((this.cur && this.cur.tempo) || curTempo() || 120);
    const g = $('#gridSel');
    return ({ q: spb, '8': spb / 2, '16': spb / 4, '8t': spb / 3 })[(g && g.value) || '16'] || spb / 4;
  },
  snap(t) { const g = this.grid(); return g > 0 ? Math.round(t / g) * g : t; },
  pushUndo() {
    this.undo.push(JSON.stringify(this.cur.notes));
    if (this.undo.length > 60) this.undo.shift();
    this.redo.length = 0;
  },
  after() {
    const d = this.cur; if (!d) return;
    d.notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
    d.note_count = d.notes.length;
    d.contour = topContour(d.notes);
    d.edited = true;
    (this.onChange || renderCommon)(d, true);
    this.postSoon();
  },
  doUndo() {
    if (!this.undo.length) return;
    this.redo.push(JSON.stringify(this.cur.notes));
    this.cur.notes = JSON.parse(this.undo.pop());
    this.sel.clear(); this.after();
  },
  doRedo() {
    if (!this.redo.length) return;
    this.undo.push(JSON.stringify(this.cur.notes));
    this.cur.notes = JSON.parse(this.redo.pop());
    this.sel.clear(); this.after();
  },
  revert() {
    if (!this.cur) return;
    this.pushUndo();
    this.cur.notes = JSON.parse(this.orig);
    this.sel.clear(); this.after();
  },
  nudge(dir, big) {
    if (!this.cur || !this.sel.size) return;
    this.pushUndo();
    const step = (big ? 12 : 1) * dir;
    this.sel.forEach(n => {
      n.pitch = Math.min(127, Math.max(0, Math.round(n.pitch) + step));
      n.name = midiName(n.pitch); n.freq = midiFreq(n.pitch);
    });
    this.after();
  },
  delSel() {
    if (!this.cur || !this.sel.size) return;
    this.pushUndo();
    this.cur.notes = this.cur.notes.filter(n => !this.sel.has(n));
    this.sel.clear(); this.after();
  },
  postSoon() { clearTimeout(this._pt); this._pt = setTimeout(() => this.post(), 450); },
  post() {
    const d = this.cur;
    if (!d || !d.job_id) return;
    const ts = curTS();
    const msg = $('#scoreMsg'); if (msg) msg.textContent = '저장 중…';
    fetch('api/edit/' + d.job_id, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        notes: d.notes.map(n => ({ start: n.start, end: n.end, pitch: Math.round(n.pitch), velocity: n.velocity || 90 })),
        tempo: curTempo(), time_sig: ts, title: d.filename || 'MusicNote',
      }),
    }).then(async r => {
      const res = await r.json();
      if (!r.ok) throw new Error(res.detail || ('HTTP ' + r.status));
      d.midi_url = res.midi_url; d.musicxml_url = res.musicxml_url; d.edited = true;
      const md = $('#midi'); if (md) { md.href = res.midi_url + '?v=' + Date.now(); md.style.display = ''; }
      const mx = $('#mxmlDl'); if (mx) { mx.href = res.musicxml_url + '?v=' + Date.now(); mx.style.display = ''; }
      if (msg) msg.textContent = '저장됨 · ' + res.note_count + '음';
    }).catch(e => { if (msg) msg.textContent = '저장 실패: ' + e.message; });
  },
};

let _rollBound = false;
function bindRoll() {
  if (_rollBound) return;
  const cv = $('#roll'); if (!cv) return;
  _rollBound = true;

  const toTP = e => {
    const r = cv.getBoundingClientRect();
    const x = (e.clientX - r.left) * (cv.width / r.width);
    const y = (e.clientY - r.top) * (cv.height / r.height);
    return { t: Math.max(0, x / ROLL.pxPerSec), p: ROLL.hi - Math.floor(y / ROLL.rh) };
  };
  const noteAt = (t, p) => {
    const list = (Editor.cur && Editor.cur.notes) || [];
    for (let i = list.length - 1; i >= 0; i--) {
      const n = list[i];
      if (Math.round(n.pitch) === p && t >= n.start - 0.012 && t <= n.end + 0.012) return n;
    }
    return null;
  };

  cv.addEventListener('pointerdown', e => {
    if (!Editor.on || !ROLL) return;
    e.preventDefault();
    try { cv.setPointerCapture(e.pointerId); } catch (_) {}
    const tp = toTP(e), n = noteAt(tp.t, tp.p);
    const draw = $('#drawMode');
    if (n) {
      const onEdge = (n.end - tp.t) * ROLL.pxPerSec < 7;
      if (!e.shiftKey && !Editor.sel.has(n)) Editor.sel.clear();
      Editor.sel.add(n);
      Editor.drag = {
        mode: onEdge ? 'resize' : 'move', t0: tp.t, p0: tp.p, pushed: false,
        snap: JSON.stringify(Editor.cur.notes),
        base: [...Editor.sel].map(m => ({ m, s: m.start, e: m.end, p: Math.round(m.pitch) })),
      };
      drawRoll(Editor.cur);
    } else if (draw && draw.classList.contains('btn-on')) {
      Editor.pushUndo();
      const g = Editor.grid();
      const nn = { start: Editor.snap(tp.t), end: Editor.snap(tp.t) + g, pitch: tp.p, name: midiName(tp.p), freq: midiFreq(tp.p), velocity: 90 };
      Editor.cur.notes.push(nn);
      Editor.sel.clear(); Editor.sel.add(nn);
      Editor.drag = { mode: 'resize', t0: tp.t, p0: tp.p, pushed: true, base: [{ m: nn, s: nn.start, e: nn.end, p: nn.pitch }] };
      Editor.after();
    } else {
      Editor.sel.clear();
      drawRoll(Editor.cur);
    }
  });

  window.addEventListener('pointermove', e => {
    const dg = Editor.drag;
    if (!dg || !ROLL) return;
    e.preventDefault();
    if (!dg.pushed) {
      Editor.undo.push(dg.snap);
      if (Editor.undo.length > 60) Editor.undo.shift();
      Editor.redo.length = 0;
      dg.pushed = true;
    }
    const tp = toTP(e), dt = tp.t - dg.t0, dp = tp.p - dg.p0;
    const g = Editor.grid();
    dg.base.forEach(b => {
      if (dg.mode === 'move') {
        b.m.start = Math.max(0, Editor.snap(b.s + dt));
        b.m.end = b.m.start + (b.e - b.s);
        b.m.pitch = Math.min(127, Math.max(0, b.p + dp));
      } else {
        b.m.end = Math.max(Editor.snap(b.e + dt), b.m.start + g * 0.5, b.m.start + 0.05);
      }
      b.m.name = midiName(b.m.pitch); b.m.freq = midiFreq(b.m.pitch);
    });
    Editor.cur.note_count = Editor.cur.notes.length;
    drawRoll(Editor.cur);
  });

  const endDrag = () => { if (!Editor.drag) return; Editor.drag = null; Editor.after(); };
  window.addEventListener('pointerup', endDrag);
  window.addEventListener('pointercancel', endDrag);

  cv.addEventListener('dblclick', e => {
    if (!Editor.on || !ROLL) return;
    const tp = toTP(e), n = noteAt(tp.t, tp.p);
    if (n) {
      Editor.pushUndo();
      const i = Editor.cur.notes.indexOf(n);
      if (i >= 0) Editor.cur.notes.splice(i, 1);
      Editor.sel.delete(n);
      Editor.after();
    }
  });
}

window.addEventListener('keydown', e => {
  if (!Editor.on || !Editor.cur) return;
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
  const k = e.key.toLowerCase();
  if (e.key === ' ') { e.preventDefault(); Player.toggle(); }
  else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); Editor.delSel(); }
  else if ((e.ctrlKey || e.metaKey) && k === 'z' && !e.shiftKey) { e.preventDefault(); Editor.doUndo(); }
  else if ((e.ctrlKey || e.metaKey) && (k === 'y' || (k === 'z' && e.shiftKey))) { e.preventDefault(); Editor.doRedo(); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); Editor.nudge(1, e.shiftKey); }
  else if (e.key === 'ArrowDown') { e.preventDefault(); Editor.nudge(-1, e.shiftKey); }
});

// ---- editing directly on the notation ----------------------------------
let _scoreBound = false;
function scoreBind() {
  if (_scoreBound) return;
  const svg = $('#scoreSvg'); if (!svg) return;
  _scoreBound = true;
  let drag = null;

  const toXY = e => {
    const r = svg.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  const srcNotesAt = (pitch, t0, t1) => (Editor.cur ? Editor.cur.notes : []).filter(
    n => Math.round(n.pitch) === pitch && n.start < t1 - 1e-3 && n.end > t0 + 1e-3);
  const hitAt = (x, y) => {
    for (const h of Score._hit)
      if (x >= h.x - 5 && x <= h.x + h.w + 5 && y >= h.y - 6 && y <= h.y + h.h + 6) return h;
    return null;
  };

  svg.addEventListener('pointerdown', e => {
    if (!Editor.on || !Editor.cur) return;
    const { x, y } = toXY(e);
    const h = hitAt(x, y);
    if (h) {
      e.preventDefault();
      try { svg.setPointerCapture(e.pointerId); } catch (_) {}
      const ns = srcNotesAt(h.pitch, h.t0, h.t1);
      if (!ns.length) return;
      if (!e.shiftKey) Editor.sel.clear();
      ns.forEach(n => Editor.sel.add(n));
      drag = { y0: y, snap: JSON.stringify(Editor.cur.notes), pushed: false,
        base: [...Editor.sel].map(n => ({ n, p: Math.round(n.pitch) })) };
      drawRoll(Editor.cur); Score.render(Editor.cur);
    } else if ($('#drawMode') && $('#drawMode').classList.contains('btn-on')) {
      const m = Score._layout.find(z => x >= z.x - 10 && x <= z.x + z.w + 10 && y >= z.y - 10 && y <= z.y + z.h + 10)
        || Score._layout[0];
      if (!m) return;
      Editor.pushUndo();
      const frac = Math.max(0, Math.min(1, (x - m.x) / m.w));
      const t = Editor.snap(m.tStart + frac * (m.tEnd - m.tStart));
      const p = Score.yToPitch(y);
      const g = Editor.grid();
      const nn = { start: t, end: t + g, pitch: p, name: midiName(p), freq: midiFreq(p), velocity: 90 };
      Editor.cur.notes.push(nn);
      Editor.sel.clear(); Editor.sel.add(nn);
      Editor.after();
    } else {
      Editor.sel.clear();
      drawRoll(Editor.cur); Score.render(Editor.cur);
    }
  });

  window.addEventListener('pointermove', e => {
    if (!drag) return;
    e.preventDefault();
    if (!drag.pushed) {
      Editor.undo.push(drag.snap);
      if (Editor.undo.length > 60) Editor.undo.shift();
      Editor.redo.length = 0;
      drag.pushed = true;
    }
    const { y } = toXY(e);
    const dSemi = Math.round((drag.y0 - y) / 6);   // ~6 px per semitone
    drag.base.forEach(b => {
      b.n.pitch = Math.min(108, Math.max(21, b.p + dSemi));
      b.n.name = midiName(b.n.pitch); b.n.freq = midiFreq(b.n.pitch);
    });
    drawRoll(Editor.cur);
  });
  const end = () => { if (!drag) return; drag = null; Editor.after(); };
  window.addEventListener('pointerup', end);
  window.addEventListener('pointercancel', end);

  svg.addEventListener('dblclick', e => {
    if (!Editor.on || !Editor.cur) return;
    const { x, y } = toXY(e);
    const h = hitAt(x, y);
    if (!h) return;
    const ns = srcNotesAt(h.pitch, h.t0, h.t1);
    if (!ns.length) return;
    Editor.pushUndo();
    Editor.cur.notes = Editor.cur.notes.filter(n => !ns.includes(n));
    ns.forEach(n => Editor.sel.delete(n));
    Editor.after();
  });
}

// ---- editor-toolbar wiring (guarded: analysis page has no toolbar) ------
function _on(sel, ev, fn) { const el = $(sel); if (el) el.addEventListener(ev, fn); }
_on('#editToggle', 'change', e => {
  Editor.on = e.target.checked;
  const bar = $('#editBar'); if (bar) bar.style.display = Editor.on ? 'flex' : 'none';
  $('#roll').classList.toggle('roll-edit', Editor.on);
  if (Editor.cur) drawRoll(Editor.cur);
});
_on('#drawMode', 'click', () => $('#drawMode').classList.toggle('btn-on'));
_on('#edUp', 'click', () => Editor.nudge(1, false));
_on('#edDown', 'click', () => Editor.nudge(-1, false));
_on('#edDel', 'click', () => Editor.delSel());
_on('#edUndo', 'click', () => Editor.doUndo());
_on('#edRedo', 'click', () => Editor.doRedo());
_on('#edRevert', 'click', () => Editor.revert());
_on('#saveScore', 'click', () => Editor.post());
_on('#keySel', 'change', () => { if (CUR) Score.render(CUR); });
_on('#tsSel', 'change', () => { if (CUR) Score.render(CUR); if (Editor.cur) Editor.postSoon(); });
_on('#scoreTempo', 'input', () => { TEMPO_TOUCHED = true; if (CUR) { CUR.tempo = curTempo(); Score.render(CUR); } });
_on('#scoreTempo', 'change', () => { if (Editor.cur) Editor.postSoon(); });
_on('#json', 'click', () => {
  const data = CUR || LAST;
  if (!data) return;
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (data.filename || 'musicnote') + '.notes.json';
  a.click();
});

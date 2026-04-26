/* ═══════════════════════════════════════
   CONSTANTS & STATE
═══════════════════════════════════════ */
const API = '';
const HISTORY_KEY = 'eatlytic_v5';
const PROFILE_KEY = 'eatlytic_profile';
const STREAK_KEY = 'eatlytic_streak';
const SAT_COLORS = ['var(--yellow)', 'var(--pink)', 'var(--mint)', 'var(--blue)', 'var(--orange)', 'var(--lilac)'];
const SAT_BG = ['#FFF8C0', '#FFE0EC', '#C0FFE8', '#C8D8FF', '#FFE8C0', '#F0C0FF'];
const CONFETTI_COLORS = ['#FFD600', '#FF2D78', '#0047FF', '#00C896', '#FF6B00', '#C084FC', '#0A0A0A'];

let state = {
  blob: null, persona: 'General Adult', language: 'en',
  lastResult: null, abortCtrl: null, quotaRemaining: 5, isPro: false,
  cameraStream: null, cameraFacing: 'environment', torchOn: false,
  camMode: 'scan',
  voiceRecog: null, isListening: false,
  conIngredients: [],
  histFilter: 'today',
  currentIngDetail: null,
};

/* ═══════════════════════════════════════
   NAVIGATION
═══════════════════════════════════════ */
function goTo(name) {
  const prev = document.querySelector('.screen.active');
  if (prev) { prev.classList.add('exit'); setTimeout(() => prev.classList.remove('active', 'exit'), 400); }
  setTimeout(() => {
    const next = document.getElementById('s-' + name);
    if (next) next.classList.add('active');
  }, 55);
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  const nb = document.getElementById('nav-' + name);
  if (nb) nb.classList.add('active');
  if (name === 'map') setTimeout(initConstellation, 280);
  if (name === 'history') setTimeout(renderHistory, 100);
  if (name === 'profile') setTimeout(renderProfile, 100);
  if (name !== 'camera' && state.cameraStream) stopCamera();
}

/* ═══════════════════════════════════════
   CHIPS
═══════════════════════════════════════ */
document.getElementById('persona-row').addEventListener('click', e => {
  const c = e.target.closest('.chip'); if (!c) return;
  document.querySelectorAll('#persona-row .chip').forEach(x => x.classList.remove('active'));
  c.classList.add('active'); state.persona = c.dataset.persona;
});
document.getElementById('lang-row').addEventListener('click', e => {
  const c = e.target.closest('.lang-chip'); if (!c) return;
  document.querySelectorAll('.lang-chip').forEach(x => x.classList.remove('active'));
  c.classList.add('active'); state.language = c.dataset.lang;
});
document.getElementById('goal-row')?.addEventListener('click', e => {
  const c = e.target.closest('.chip'); if (!c) return;
  document.querySelectorAll('#goal-row .chip').forEach(x => x.classList.remove('active'));
  c.classList.add('active');
});

/* ═══════════════════════════════════════
   CAMERA — LIVE VIEWFINDER
═══════════════════════════════════════ */
async function openCamera() {
  goTo('camera');
  await startCameraStream();
}

async function startCameraStream() {
  stopCamera();
  try {
    const constraints = { video: { facingMode: state.cameraFacing, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.cameraStream = stream;
    const video = document.getElementById('camera-video');
    video.srcObject = stream;
    await video.play();
    showToast('📷 Camera active');
  } catch (err) {
    showToast('⚠️ Camera unavailable: ' + (err.message || err.name));
    goTo('scan');
  }
}

function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }
  const video = document.getElementById('camera-video');
  if (video) { video.srcObject = null; }
}

async function flipCamera() {
  state.cameraFacing = state.cameraFacing === 'environment' ? 'user' : 'environment';
  await startCameraStream();
}

async function toggleTorch() {
  if (!state.cameraStream) return;
  const track = state.cameraStream.getVideoTracks()[0];
  const caps = track.getCapabilities?.() || {};
  if (!caps.torch) { showToast('Torch not supported'); return; }
  state.torchOn = !state.torchOn;
  await track.applyConstraints({ advanced: [{ torch: state.torchOn }] });
  document.getElementById('torch-btn').classList.toggle('on', state.torchOn);
}

function setCamMode(mode) {
  state.camMode = 'scan';
  document.querySelectorAll('.cam-mode-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('cam-mode-scan')?.classList.add('active');
  document.getElementById('camera-reticle').style.display = '';
  document.getElementById('cam-overlay-pill').textContent = 'Point at a food label';
}

function captureFrame() {
  const video = document.getElementById('camera-video');
  const canvas = document.getElementById('capture-canvas');
  if (!video.readyState || video.readyState < 2) { showToast('Camera not ready'); return; }
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(blob => {
    if (!blob) { showToast('Capture failed'); return; }
    stopCamera();
    processFile(blob);
    goTo('scan');
    const fl = document.getElementById('scan-flash');
    fl.classList.add('flash'); setTimeout(() => fl.classList.remove('flash'), 130);
    spawnConfetti(window.innerWidth / 2, window.innerHeight / 2);
  }, 'image/jpeg', 0.93);
}

/* ── FILE UPLOAD */
function handlePortalTap() { if (!state.blob) openCamera(); }
function triggerUpload() { document.getElementById('file-input').click(); }
function handleFile(e) { const f = e.target.files?.[0]; if (f) processFile(f); e.target.value = ''; }
function onDragOver(e) { e.preventDefault(); document.getElementById('portal-core').style.transform = 'scale(1.05)'; }
function onDragLeave() { document.getElementById('portal-core').style.transform = ''; }
function onDrop(e) {
  e.preventDefault(); document.getElementById('portal-core').style.transform = '';
  const f = e.dataTransfer.files?.[0];
  if (f && f.type.startsWith('image/')) processFile(f);
  else showToast('⚠️ Drop an image file');
}

function processFile(file) {
  state.blob = file;
  const url = URL.createObjectURL(file);
  const img = document.getElementById('preview-img');
  img.onload = null;
  img.src = url;
  const wrap = document.getElementById('portal-preview-wrap');
  wrap.classList.add('visible');
  document.getElementById('preview-badge').textContent = '✓ Ready to analyse';
  document.getElementById('analyse-cta').classList.add('visible');
  document.getElementById('analyse-cta').disabled = false;
  document.getElementById('analyse-cta').textContent = '✦ Analyse this Label →';

  const fl = document.getElementById('scan-flash');
  fl.classList.add('flash'); setTimeout(() => fl.classList.remove('flash'), 120);
  spawnConfetti(window.innerWidth / 2, window.innerHeight * 0.38);

  checkQuality(file);
}

async function checkQuality(file) {
  document.getElementById('quality-strip').classList.add('visible');
  try {
    const fd = new FormData(); fd.append('image', file);
    const res = await fetch(`${API}/check-image`, { method: 'POST', body: fd });
    if (!res.ok) { setQualityLocal(file); return; }
    const d = await res.json();
    applyQuality(d.blur_severity || 'none', Math.max(5, Math.min(100, 100 - (d.blur_score || 50))));
  } catch { setQualityLocal(file); }
}

async function setQualityLocal(file) {
  const bmp = await createImageBitmap(file);
  const c = document.createElement('canvas');
  c.width = Math.min(200, bmp.width); c.height = Math.min(200, bmp.height);
  const ctx = c.getContext('2d');
  ctx.drawImage(bmp, 0, 0, c.width, c.height);
  const px = ctx.getImageData(0, 0, c.width, c.height).data;
  let lap = 0;
  for (let i = 4; i < px.length - 4; i += 4) lap += Math.abs(px[i] - px[i + 4]) + Math.abs(px[i] - px[i - 4]);
  const score = Math.min(100, Math.round(lap / (c.width * c.height) * 2));
  applyQuality(score > 60 ? 'none' : score > 30 ? 'mild' : 'blurry', score);
}

async function applyQuality(sev, pct) {
  const fill = document.getElementById('qs-fill'), badge = document.getElementById('qs-badge'), label = document.getElementById('qs-label');
  let d = sev === 'none' ? Math.max(92, pct) : sev === 'mild' ? Math.max(60, pct) : Math.max(15, pct);
  fill.style.width = d + '%';

  const cta = document.getElementById('analyse-cta');

  if (sev === 'none') {
    fill.style.background = 'var(--mint)'; badge.style.color = 'var(--mint)'; badge.textContent = '✓ Clear'; label.textContent = 'Great quality';
    cta.textContent = '✦ Analyse this Label →';
    cta.disabled = false;
  }
  else {
    // Blurry or Mild - Trigger Auto-Enhance in background
    fill.style.background = (sev === 'mild') ? 'var(--orange)' : 'var(--red)';
    badge.style.color = (sev === 'mild') ? 'var(--orange)' : 'var(--red)';
    badge.textContent = (sev === 'mild') ? '~ Fair' : '⚠ Blurry';
    label.textContent = 'Enhance suggests…';

    cta.textContent = '✦ Analyse Anyway →';
    cta.disabled = false;

    // Auto-enhance flow (non-blocking)
    try {
      const fd = new FormData(); fd.append('image', state.blob);
      fetch(`${API}/enhance-preview`, { method: 'POST', body: fd }).then(res => {
        if (res.ok) return res.json();
      }).then(data => {
        if (data && data.deblurred && data.image_b64) {
          const rawB64 = data.image_b64.includes(',') ? data.image_b64.split(',')[1] : data.image_b64;
          const newUrl = data.image_b64.includes(',') ? data.image_b64 : `data:image/jpeg;base64,${data.image_b64}`;
          document.getElementById('preview-img').src = newUrl;
          const byteChars = atob(rawB64);
          const byteNums = new Array(byteChars.length);
          for (let i = 0; i < byteChars.length; i++) byteNums[i] = byteChars.charCodeAt(i);
          const byteArray = new Uint8Array(byteNums);
          state.blob = new Blob([byteArray], { type: 'image/jpeg' });

          label.textContent = 'Fixed by AI';
          badge.textContent = '✓ Enhanced';
          badge.style.color = 'var(--mint)';
          fill.style.background = 'var(--mint)';
          fill.style.width = '100%';
          cta.textContent = '✦ Analyse Enhanced Label →';
          showToast('✨ Image sharpened!');
        }
      }).catch(() => { });
    } catch (err) { }
  }
}

/* ═══════════════════════════════════════
   VOICE LOGGING
═══════════════════════════════════════ */
function setupVoice() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) { document.getElementById('voice-btn').style.display = 'none'; return; }
  const recog = new SpeechRec();
  recog.lang = 'en-US'; recog.continuous = false; recog.interimResults = true; recog.maxAlternatives = 1;
  recog.onresult = e => {
    let t = '';
    for (let i = e.resultIndex; i < e.results.length; i++) t += e.results[i][0].transcript;
    document.getElementById('voice-transcript').textContent = t;
    if (e.results[e.resultIndex]?.isFinal) parseVoiceMeal(t);
  };
  recog.onerror = e => { showToast('Voice error: ' + e.error); stopVoice(); };
  recog.onend = () => stopVoice();
  state.voiceRecog = recog;
}

function toggleVoice() { state.isListening ? stopVoice() : startVoice(); }
function startVoice() {
  if (!state.voiceRecog) { showToast('Voice not supported'); return; }
  state.isListening = true;
  document.getElementById('voice-btn').classList.add('listening');
  document.getElementById('voice-icon').textContent = '🔴';
  document.getElementById('voice-transcript').textContent = 'Listening…';
  try { state.voiceRecog.start(); } catch { }
}
function stopVoice() {
  state.isListening = false;
  document.getElementById('voice-btn').classList.remove('listening');
  document.getElementById('voice-icon').textContent = '🎤';
  try { state.voiceRecog?.stop(); } catch { }
}

async function parseVoiceMeal(transcript) {
  stopVoice();
  document.getElementById('voice-transcript').textContent = '🧠 Parsing "' + transcript + '"…';
  try {
    const fd = new FormData(); fd.append('text', transcript); fd.append('persona', state.persona); fd.append('language', state.language);
    const res = await fetch(`${API}/parse-voice-meal`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error('API unavailable');
    const d = await res.json();
    if (d.error) throw new Error(d.message || d.error);
    state.lastResult = d; saveHistory(d, null); buildConstellationData(d); renderReveal(d); goTo('reveal');
    spawnConfetti(window.innerWidth / 2, window.innerHeight / 2);
    document.getElementById('voice-transcript').textContent = 'Say what you ate…';
  } catch {
    const fakeResult = {
      product_name: transcript, score: 5, verdict: 'Voice logged',
      summary: 'Logged via voice: "' + transcript + '". Analysis requires image scan.',
      nutrient_breakdown: [], pros: [], cons: [], ingredients_spotlight: [], is_voice_log: true
    };
    state.lastResult = fakeResult; saveHistory(fakeResult, null); renderReveal(fakeResult); goTo('reveal');
    document.getElementById('voice-transcript').textContent = '✓ Logged: ' + transcript;
    showToast('✓ Voice meal logged!'); spawnStickerPop('🎤', window.innerWidth / 2, window.innerHeight / 2);
  }
}

/* ═══════════════════════════════════════
   VINE LOADER
═══════════════════════════════════════ */
const VINE_STEPS = [
  { label: 'Clarity Scan', pct: 20 }, { label: 'OCR Extraction', pct: 40 },
  { label: 'Lie Detection', pct: 60 }, { label: 'Scoring Engine', pct: 85 }, { label: 'Final Verdict', pct: 100 }
];

function startVine() {
  const fill = document.getElementById('vine-fill'), nodes = document.getElementById('vine-nodes'), txt = document.getElementById('reading-text');
  fill.style.height = '0%'; nodes.innerHTML = ''; txt.innerHTML = '';
  VINE_STEPS.forEach((step, i) => {
    const top = (1 - step.pct / 100) * 100;
    const nd = document.createElement('div'); nd.className = 'vine-node'; nd.id = 'vn-' + i; nd.style.top = top + '%';
    const lbl = document.createElement('div'); lbl.className = 'vine-node-label'; lbl.textContent = step.label;
    nd.appendChild(lbl); nodes.appendChild(nd);
  });
  let step = 0;
  function advance() {
    if (step >= VINE_STEPS.length) return;
    fill.style.height = VINE_STEPS[step].pct + '%';
    document.getElementById('vn-' + step)?.classList.add('lit');
    step++; setTimeout(advance, step >= VINE_STEPS.length ? 400 : 500);
  }
  setTimeout(advance, 300);
}

function crystallizeText(text) {
  const el = document.getElementById('reading-text'); el.innerHTML = '';
  [...text].forEach((ch, i) => {
    const s = document.createElement('span'); s.className = 'char'; s.textContent = ch;
    s.style.animationDelay = (i * 18) + 'ms'; el.appendChild(s);
  });
}

/* ═══════════════════════════════════════
   RUN ANALYSIS
═══════════════════════════════════════ */
async function runAnalysis() {
  if (!state.blob) return;
  if (state.quotaRemaining <= 0 && !state.isPro) { openPaywall('limit'); return; }
  goTo('reading'); startVine();
  state.abortCtrl = new AbortController();
  const tid = setTimeout(() => state.abortCtrl?.abort(), 90000);

  let thumbData = null;
  try {
    const bmp = await createImageBitmap(state.blob);
    const tc = document.createElement('canvas'); tc.width = 80; tc.height = 80;
    const tctx = tc.getContext('2d'), ar = bmp.width / bmp.height;
    if (ar > 1) tctx.drawImage(bmp, (bmp.width - bmp.height) / 2, 0, bmp.height, bmp.height, 0, 0, 80, 80);
    else tctx.drawImage(bmp, 0, (bmp.height - bmp.width) / 2, bmp.width, bmp.width, 0, 0, 80, 80);
    thumbData = tc.toDataURL('image/jpeg', 0.6);
  } catch { }

  const fd = new FormData();
  fd.append('image', state.blob); fd.append('persona', state.persona);
  fd.append('language', state.language); fd.append('age_group', 'adult'); fd.append('product_category', 'general');

  try {
    const res = await fetch(`${API}/analyze`, { method: 'POST', body: fd, signal: state.abortCtrl.signal });
    clearTimeout(tid);
    const data = await res.json();
    if (res.status === 429 || data.error === 'quota_exceeded') { goTo('scan'); openPaywall('limit'); return; }
    if (data.error) {
      const errType = data.error === 'no_label' && data.tip === 'wrong_side' ? 'wrong_side' : data.error;
      goTo('scan'); showError(errType, data.message || 'Analysis failed.'); return;
    }
    if (data.scan_meta) { state.quotaRemaining = data.scan_meta.scans_remaining ?? state.quotaRemaining; state.isPro = data.scan_meta.is_pro ?? state.isPro; updateQuota(); }
    const sampleText = data.extracted_text
      ? data.extracted_text.substring(0, 80) + '…'
      : 'Ingredients: ' + (data.ingredients_spotlight?.map(i => i.name) || []).slice(0, 5).join(', ') + '…';
    crystallizeText(sampleText);
    if (!data.risk_flags) buildRiskFlagsFromAnalysis(data);
    state.lastResult = data; saveHistory(data, thumbData); buildConstellationData(data); updateStreak();
    setTimeout(() => { renderReveal(data); goTo('reveal'); resetPortal(); }, 2200);
  } catch (e) {
    clearTimeout(tid); goTo('scan');
    if (e.name === 'AbortError') showToast('⏱ Analysis cancelled');
    else showError('error', e.message || 'Network error.');
  }
}

function buildRiskFlagsFromAnalysis(data) {
  const flags = [], nutr = data.nutrient_breakdown || [], ingrs = data.ingredients_spotlight || [];
  const sodium = nutr.find(n => n.name?.toLowerCase().includes('sodium'))?.value || 0;
  const sugar = nutr.find(n => n.name?.toLowerCase().includes('sugar'))?.value || 0;
  const addrs = ingrs.filter(i => i.type === 'additive' || i.type === 'preservative');
  if (addrs.length > 0) flags.push({ label: 'Additives (' + addrs.length + ')', level: 'orange', icon: '⚗️' });
  if (sugar > 15) flags.push({ label: 'High sugar', level: 'red', icon: '🍬' });
  if (sodium > 700) flags.push({ label: 'High sodium', level: 'orange', icon: '🧂' });
  if (ingrs.some(i => ['e102', 'e110', 'e122', 'tartrazine'].some(k => i.name?.toLowerCase().includes(k))))
    flags.push({ label: 'Artificial dye', level: 'red', icon: '🎨' });
  if (data.score >= 8) flags.push({ label: 'Clean label', level: 'green', icon: '✓' });
  data.risk_flags = flags;
}

function cancelAnalysis() { state.abortCtrl?.abort(); goTo('scan'); }

function showError(type, message) {
  const icon = type === 'no_label' ? '🔄' : type === 'wrong_side' ? '🔄' : type === 'no_text' ? '🔍' : type === 'blurry_image' ? '📐' : type === 'invalid_image' ? '🖼' : '❌';
  const title = type === 'no_label' ? 'No Nutrition Label Found' : type === 'wrong_side' ? "That's the Front — Flip It Over!" : type === 'no_text' ? 'No Text Detected' : type === 'blurry_image' ? 'Move Closer to the Label' : type === 'invalid_image' ? 'Invalid Image' : 'Analysis Failed';
  const extra = (type === 'no_label' || type === 'wrong_side' || type === 'no_text')
    ? `<div style="margin:20px auto;max-width:250px;display:flex;align-items:center;justify-content:center;gap:15px;background:rgba(255,45,120,0.05);padding:15px;border-radius:12px;border:1.5px dashed var(--pink)">
    <div style="display:flex;flex-direction:column;align-items:center;gap:4px"><div style="width:40px;height:55px;background:var(--pink);border:2px solid var(--ink);border-radius:4px;display:flex;align-items:center;justify-content:center;color:white;font-weight:900;font-size:10px">FRONT</div><span style="font-size:12px">❌</span></div>
    <div style="font-size:1.5rem;animation:portal-breath 2s infinite">🔄</div>
    <div style="display:flex;flex-direction:column;align-items:center;gap:4px"><div style="width:40px;height:55px;background:var(--mint);border:2px solid var(--ink);border-radius:4px;display:flex;flex-direction:column;align-items:center;padding:4px;gap:2px"><div style="width:100%;height:3px;background:white"></div><div style="width:100%;height:3px;background:white"></div><div style="width:100%;height:3px;background:white"></div><div style="width:100%;height:8px;background:rgba(0,0,0,0.1)"></div></div><span style="font-size:12px">✅</span></div>
   </div>
   <div style="font-size:12px;font-weight:700;color:var(--ink);margin-bottom:8px">Flip the product! We need the <span style="color:var(--mint)">Back Label</span> (showing ingredients &amp; nutrients) to analyze it.</div>`
    : '';
  document.getElementById('reveal-scroll').innerHTML = `<div class="result-empty"><div class="result-empty-icon">${icon}</div><div class="result-empty-title">${title}</div>${extra}<div class="result-empty-sub">${esc(message)}</div><button class="result-empty-btn" onclick="goTo('scan')">Try Again →</button></div>`;
  goTo('reveal');
}

/* ═══════════════════════════════════════
   RENDER REVEAL
═══════════════════════════════════════ */
function renderReveal(d) {
  const score = d.score ?? 0;
  const safetyTier = (d.safety_tier || 'Limit').toLowerCase();
  const safetyColor = safetyTier === 'safe' ? 'var(--mint)' : safetyTier === 'limit' ? 'var(--orange)' : 'var(--red)';
  const safetyIcon = safetyTier === 'safe' ? '🟢' : safetyTier === 'limit' ? '🟡' : '🔴';

  let html = '';

  // --- HERO SAFETY VERDICT (The Wedge) ---
  html += `
  <div class="safety-verdict-box" style="border-color:${safetyColor}">
      <div class="sv-header" style="color:${safetyColor}">
          <span class="sv-icon">${safetyIcon}</span>
          <span class="sv-status">${esc(d.safety_verdict || d.safety_tier || 'Review Recommended')}</span>
      </div>
      <div class="sv-reason">${esc(d.safety_reason || 'Scan complete. Review safety details below.')}</div>
  </div>`;

  html += `<div class="reveal-top"><div class="reveal-product">${esc(d.product_name || 'Unknown Product')}</div><div class="reveal-score-group"><span class="reveal-score" style="color:${safetyColor}">${score}</span><span class="reveal-verdict-label" style="color:${safetyColor}">${esc(d.verdict || '')}</span></div></div>`;

  const meta = [d.product_category && esc(d.product_category), d.nutriscore && 'Nutri-Score: ' + esc(d.nutriscore), d.nova_group && 'NOVA: ' + d.nova_group, d.is_barcode_scan && 'Barcode scan', d.is_voice_log && 'Voice logged'].filter(Boolean);

  const perfHtml = d.perf_metrics?.latency_ms ? `<div class="perf-badge">⚡️ ${(d.perf_metrics.latency_ms / 1000).toFixed(1)}s</div>` : '';

  if (meta.length || perfHtml) {
    html += `
    <div class="confidence-strip">
        <div class="conf-dot"></div>
        <span style="font-size:9px;letter-spacing:1px">${meta.join(' · ')}</span>
        <div style="flex-grow:1"></div>
        ${perfHtml}
    </div>`;
  }

  html += `<div class="result-actions"><button class="rac-btn" onclick="shareResult()">📤 Share</button><button class="rac-btn secondary" onclick="goTo('map')">✦ Map</button><button class="rac-btn secondary" onclick="openDuelSelection(${d.scan_id})">⚔️ Duel</button></div>`;

  if (d.risk_flags?.length) {
    html += `<div class="risk-flags">`;
    d.risk_flags.forEach(f => { html += `<div class="risk-flag rf-${f.level}"><div class="rf-dot"></div>${esc(f.icon)} ${esc(f.label)}</div>`; });
    html += `</div>`;
  }

  html += `<div class="orb-stage"><div class="score-orb-wrap" id="sorb-wrap"><div class="orb-pulse-ring" style="border-color:${safetyColor}"></div><div class="score-orb" onclick="expandSatellite('summary')"><div class="orb-num" style="color:${safetyColor}">${score}</div><div class="orb-denom">/10</div></div></div></div>`;

  if (d.blur_info?.detected) html += `<div class="blur-notice visible">${d.blur_info.deblurred ? `<strong>🔧 Enhanced</strong> — ${d.blur_info.method_log || 'AI deblur'}.` : `<strong>📷 Blur detected</strong> (${d.blur_info.severity}) — results may vary.`}</div>`;

  if (d.nutrient_breakdown?.length) {
    html += `<div class="pack-label">Your Nutrient Pack — ${d.nutrient_breakdown.length} cards</div><div class="nutrient-pack" id="nutrient-pack">`;
    d.nutrient_breakdown.forEach((n, i) => {
      const bg = SAT_BG[i % SAT_BG.length], rot = ((i % 2 === 0 ? 1 : -1) * (i % 3 + 1)) + 'deg';
      const rIcon = { good: '✓', moderate: '~', caution: '~', bad: '!' }[n.rating] || '?';
      html += `<div class="ns-card" id="nsc-${i}" style="background:${bg};--rot:${rot}" onclick="flipNutrientCard(${i})"><div class="ns-rating-badge">${rIcon}</div><span class="ns-icon">${getNutrientIcon(n.name)}</span><div class="ns-name">${esc(n.name)}</div><div class="ns-val">${n.value}<small style="font-size:0.55em;font-weight:700"> ${n.unit || ''}</small></div><div class="ns-bar"><div class="ns-fill" id="nsf-${i}"></div></div></div>`;
    });
    html += `</div>`;
  }

  const glyph = score >= 7 ? '✓' : score >= 4 ? '~' : '!';
  if (d.summary) html += `<div class="verdict-tape"><div class="vt-stamp" style="background:${score >= 7 ? 'var(--mint)' : score >= 4 ? 'var(--yellow)' : 'var(--pink)'}">${glyph}</div><div><div class="vt-title">${esc(d.verdict || '')}</div><div class="vt-desc">${esc(d.summary)}</div></div></div>`;

  if (d.ingredients_spotlight?.length) {
    html += `<div class="ingr-section"><div class="section-label" style="margin-bottom:8px">Ingredients spotted</div><div class="ingr-tags">`;
    d.ingredients_spotlight.forEach(ing => {
      const t = ing.type?.toLowerCase();
      const cls = (t === 'additive' || t === 'preservative') ? 'itag-warn' : (ing.safety_rating?.toLowerCase() === 'safe' || t === 'natural' || t === 'vitamin') ? 'itag-good' : 'itag-ok';
      html += `<div class="ingr-tag ${cls}" onclick="showIngDetail(${JSON.stringify(ing).replace(/"/g, '&quot;')})">${esc(ing.name)}</div>`;
    });
    html += `</div></div>`;
  }

  if (d.pros?.length || d.cons?.length) {
    html += `<div class="pro-con-grid">`;
    if (d.pros?.length) { html += `<div class="pc-card"><div class="pc-head pros">Benefits</div>`; d.pros.forEach(p => { html += `<div class="pc-item">${esc(p)}</div>`; }); html += `</div>`; }
    if (d.cons?.length) { html += `<div class="pc-card"><div class="pc-head cons">Concerns</div>`; d.cons.forEach(c => { html += `<div class="pc-item">${esc(c)}</div>`; }); html += `</div>`; }
    html += `</div>`;
  }

  if (d.age_warnings?.length) {
    html += `<div class="section-label" style="padding:4px 0 8px">Who should be cautious?</div><div class="age-grid">`;
    d.age_warnings.forEach(w => { html += `<div class="age-card ${w.status || 'caution'}"><div class="ac-group">${w.emoji || ''} ${esc(w.group)}</div><div class="ac-msg">${esc(w.message)}</div></div>`; });
    html += `</div>`;
  }

  if (d.summary && (d.eli5_explanation || d.molecular_insight)) {
    html += `<div class="insight-tabs">${d.summary ? `<button class="itab active" onclick="switchInsight(event,'is0')">Summary</button>` : ''} ${d.eli5_explanation ? `<button class="itab" onclick="switchInsight(event,'is1')">Simple</button>` : ''} ${d.molecular_insight ? `<button class="itab" onclick="switchInsight(event,'is2')">Science</button>` : ''}</div>
${d.summary ? `<div class="insight-panel active" id="is0"><div class="insight-text">${esc(d.summary)}</div></div>` : ''}
${d.eli5_explanation ? `<div class="insight-panel" id="is1"><div class="insight-text">${esc(d.eli5_explanation)}</div></div>` : ''}
${d.molecular_insight ? `<div class="insight-panel" id="is2"><div class="insight-text">${esc(d.molecular_insight)}</div></div>` : ''}`;
  }

  if (d.better_alternative) html += `<div class="alt-card"><div style="font-size:1.2rem">💡</div><div><div class="alt-label">Better Alternative</div><div class="alt-text">${esc(d.better_alternative)}</div></div></div>`;

  // --- FEEDBACK LOOP (Phase 2 hardening) ---
  if (d.scan_id) {
    html += `<button class="report-error-btn" onclick="reportScanError(${d.scan_id})">🚩 Result inaccurate? Report for review</button>`;
  }

  document.getElementById('reveal-scroll').innerHTML = html;
  buildSatellites(d);

  requestAnimationFrame(() => {
    document.querySelectorAll('.ns-card').forEach((card, i) => {
      setTimeout(() => {
        card.classList.add('visible');
        const fill = document.getElementById('nsf-' + i);
        if (!fill) return;
        const n = d.nutrient_breakdown?.[i];
        if (n) { const pct = Math.min(100, Math.round(((n.value || 0) / (n.unit === 'g' ? 50 : n.unit === 'mg' ? 2400 : 100)) * 100)); setTimeout(() => { fill.style.width = pct + '%'; }, 500); }
      }, i * 130);
    });
  });
}

function showIngDetail(ing) {
  if (typeof ing === 'string') { try { ing = JSON.parse(ing); } catch { return; } }
  const risk = ing.type?.toLowerCase();
  const rc = { additive: '#FF2D78', preservative: '#FF2D78', natural: '#00A878', vitamin: '#C084FC' }[risk] || '#FF6B00';
  const rl = { additive: 'Additive', preservative: 'Preservative', natural: 'Natural', vitamin: 'Vitamin', emulsifier: 'Emulsifier', seasoning: 'Seasoning' }[risk] || (risk || 'Ingredient');
  document.getElementById('ing-detail-name').textContent = ing.name || 'Ingredient';
  const riskEl = document.getElementById('ing-detail-risk');
  riskEl.style.cssText = `background:${rc}18;border-color:${rc};color:${rc}`;
  riskEl.textContent = rl;
  document.getElementById('ing-detail-what').textContent = ing.what_it_is || 'An ingredient found in this product.';
  document.getElementById('ing-detail-impact').textContent = ing.health_impact || 'Effects vary based on quantity consumed.';
  document.getElementById('ing-detail-fact').textContent = ing.curiosity_fact || ing.health_impact || 'Tap the Constellation Map for more details.';
  document.getElementById('ing-detail-overlay').classList.add('open');
}
function closeIngDetail() { document.getElementById('ing-detail-overlay').classList.remove('open'); }

function buildSatellites(d) {
  const wrap = document.getElementById('sorb-wrap'); if (!wrap) return;
  wrap.querySelectorAll('.orb-satellite').forEach(e => e.remove());
  const sats = (d.nutrient_breakdown || []).slice(0, 6).map((n, i) => ({
    label: n.name.split(' ')[0], val: n.value + (n.unit || ''),
    color: SAT_COLORS[i % SAT_COLORS.length], bg: SAT_BG[i % SAT_BG.length],
    desc: n.impact || `${n.name}: ${n.value}${n.unit || ''} in this product.`, full: n,
  }));
  sats.forEach((sat, i) => {
    const el = document.createElement('div'); el.className = 'orb-satellite sat-' + i;
    el.innerHTML = `<div class="sat-dot" style="background:${sat.bg}"><span style="font-size:9px;font-weight:800;color:var(--ink)">${sat.val}</span></div><div class="sat-label">${esc(sat.label)}</div>`;
    el.onclick = () => expandSatelliteData(sat); wrap.appendChild(el);
  });
}

function expandSatelliteData(sat) {
  document.getElementById('se-name').textContent = sat.full?.name || sat.label;
  document.getElementById('se-val').textContent = sat.val; document.getElementById('se-val').style.color = sat.color;
  document.getElementById('se-desc').textContent = sat.desc;
  document.getElementById('sat-expand').classList.add('open');
}
function expandSatellite(key) {
  if (key === 'summary' && state.lastResult) {
    document.getElementById('se-name').textContent = state.lastResult.product_name || 'Product';
    document.getElementById('se-val').textContent = state.lastResult.score + '/10'; document.getElementById('se-val').style.color = 'var(--ink)';
    document.getElementById('se-desc').textContent = state.lastResult.summary || '';
    document.getElementById('sat-expand').classList.add('open');
  }
}
function closeSatellite() { document.getElementById('sat-expand').classList.remove('open'); }

function flipNutrientCard(i) {
  const card = document.getElementById('nsc-' + i), n = state.lastResult?.nutrient_breakdown?.[i];
  if (!card || !n) return;
  spawnStickerPop(getNutrientIcon(n.name), card.getBoundingClientRect().left + 20, card.getBoundingClientRect().top);
  showToast(getNutrientIcon(n.name) + ' ' + n.name + ': ' + (n.impact || n.value + (n.unit || '')));
  card.style.transform = 'scale(1.18) rotate(0deg)';
  setTimeout(() => { if (card.classList.contains('visible')) card.style.transform = ''; }, 500);
}
function switchInsight(e, panel) {
  document.querySelectorAll('.itab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.insight-panel').forEach(p => p.classList.remove('active'));
  e.target.classList.add('active'); document.getElementById(panel)?.classList.add('active');
}
function getNutrientIcon(name) {
  const n = name?.toLowerCase() || '';
  if (n.includes('sodium') || n.includes('salt')) return '🧂';
  if (n.includes('protein')) return '💪';
  if (n.includes('calorie') || n.includes('energy')) return '⚡';
  if (n.includes('carb') || n.includes('sugar')) return '🌾';
  if (n.includes('fat')) return '🫒';
  if (n.includes('fibre') || n.includes('fiber')) return '🌱';
  if (n.includes('vitamin') || n.includes('calcium') || n.includes('iron')) return '💊';
  return '🔬';
}

/* ═══════════════════════════════════════
   HISTORY SCREEN
═══════════════════════════════════════ */
async function renderHistory() {
  const histContainer = document.getElementById('history-container');
  const emptyBox = document.getElementById('history-empty');
  histContainer.innerHTML = '<div style="text-align:center;padding:20px;font-size:12px;color:var(--muted)">Syncing...</div>';

  try {
    const res = await fetch(`${API}/api/v1/history`);
    const serverHist = await res.ok ? await res.json() : [];

    if (serverHist.length === 0) {
      histContainer.innerHTML = '';
      emptyBox.style.display = 'block';
      return;
    }

    emptyBox.style.display = 'none';
    let html = '';

    serverHist.forEach(item => {
      const score = item.score || 0;
      const sc = score >= 7 ? 'var(--mint)' : score >= 4 ? 'var(--orange)' : 'var(--red)';
      const date = new Date(item.scanned_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });

      html += `
        <div class="history-card" onclick="openDuelSelection(${item.id})">
            <div class="history-info">
                <div class="history-name">${esc(item.product_name)}</div>
                <div class="history-meta">${date} · ${esc(item.brand || 'No Brand')} · ${item.category || 'Food'}</div>
            </div>
            <div class="history-score" style="border-color:${sc}; color:${sc}">${score}</div>
        </div>`;
    });

    histContainer.innerHTML = html;
  } catch (e) {
    histContainer.innerHTML = '<div style="text-align:center;color:var(--red);padding:20px">Failed to load history</div>';
  }
}

/* ═══════════════════════════════════════
   DUEL MODE LOGIC
═══════════════════════════════════════ */
let duelSelection = null;

function openDuelSelection(scanId) {
  // If we don't have a first item, save this one and go to history to pick second
  if (!duelSelection) {
    duelSelection = scanId;
    showToast('⚔️ First item selected! Pick a competitor from History.');
    goTo('history');
  } else {
    // We have both items, run the duel
    if (duelSelection === scanId) {
      showToast('Choose a different product to compare!');
      return;
    }
    runDuel(duelSelection, scanId);
    duelSelection = null; // reset
  }
}

async function runDuel(idA, idB) {
  goTo('reading'); // Show loader
  crystallizeText('Simulating Duel...');

  try {
    const fd = new FormData();
    fd.append('scan_a_id', idA);
    fd.append('scan_b_id', idB);

    const res = await fetch(`${API}/api/v1/duel`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Duel simulation failed');

    const data = await res.json();
    renderDuel(data);
    goTo('duel');
    spawnConfetti(window.innerWidth / 2, 200);
  } catch (e) {
    showToast('⚠️ Oops: ' + e.message);
    goTo('history');
  }
}

function renderDuel(d) {
  const sideA = document.getElementById('duel-side-a');
  const sideB = document.getElementById('duel-side-b');
  const winnerBox = document.getElementById('duel-winner-box');

  winnerBox.innerHTML = `<div style="font-size:0.9rem; margin-bottom:5px">🏆 WINNER: <strong>${esc(d.winner_name)}</strong></div><div style="font-size:0.75rem; color:var(--muted)">${esc(d.edge)}</div>`;

  const renderSide = (prod, winnerId) => {
    const isWinner = prod.id === winnerId;
    const sc = prod.score >= 7 ? 'var(--mint)' : prod.score >= 4 ? 'var(--orange)' : 'var(--red)';

    let html = `
            ${isWinner ? '<div class="duel-winner-badge">WINNER</div>' : ''}
            <div class="duel-product-name">${esc(prod.name)}</div>
            <div class="duel-score-orb" style="border-color:${sc}; color:${sc}">${prod.score}</div>
            <div class="duel-metrics">
        `;

    d.comparison.forEach(m => {
      const winClass = (m.winner === (prod.id === d.side_by_side.a.id ? 'a' : 'b')) ? 'win' : '';
      const val = prod.id === d.side_by_side.a.id ? m.val_a : m.val_b;
      html += `
                <div class="duel-metric-row ${winClass}">
                    <span>${m.metric}</span>
                    <span class="duel-metric-val">${val}${m.unit}</span>
                </div>
            `;
    });

    html += `</div>`;
    return html;
  };

  sideA.innerHTML = renderSide(d.side_by_side.a, d.winner_id);
  sideB.innerHTML = renderSide(d.side_by_side.b, d.winner_id);
}

function setHistFilter(f) { state.histFilter = f; renderHistory(); }
function replayHistoryItem(id) {
  const item = getHistory().find(h => String(h.id) === String(id)); if (!item) return;
  state.lastResult = item.data; buildConstellationData(item.data); renderReveal(item.data); goTo('reveal');
}
function computeBadges(hist, streak) {
  return [
    { icon: '🌱', name: 'First Scan', color: '#C0FFE8', cond: 'Complete your first scan', locked: hist.length < 1 },
    { icon: '🔥', name: '3-Day Streak', color: '#FFE8C0', cond: 'Scan 3 days in a row', locked: streak.count < 3 },
    { icon: '🏆', name: '10 Scans', color: '#FFF8C0', cond: 'Complete 10 total scans', locked: hist.length < 10 },
    { icon: '💯', name: 'Perfect Score', color: '#C0FFE8', cond: 'Get a 10/10 scan', locked: !hist.some(h => h.score >= 10) },
    { icon: '🥗', name: 'Clean Eater', color: '#C0FFE8', cond: 'Get 5 scans scoring 8+', locked: hist.filter(h => h.score >= 8).length < 5 },
    { icon: '🔬', name: 'Scientist', color: '#F0C0FF', cond: 'View 5 constellation maps', locked: false },
    { icon: '🌍', name: 'Multilingual', color: '#C8D8FF', cond: 'Scan in another language', locked: hist.every(h => !h.language || h.language === 'en') },
  ];
}

function getStreak() { try { return JSON.parse(localStorage.getItem(STREAK_KEY) || '{"count":0,"lastDate":""}'); } catch { return { count: 0, lastDate: '' }; } }
function updateStreak() {
  const streak = getStreak(), today = new Date().toDateString();
  if (streak.lastDate === today) return;
  const yesterday = new Date(); yesterday.setDate(yesterday.getDate() - 1);
  streak.count = streak.lastDate === yesterday.toDateString() ? streak.count + 1 : 1;
  streak.lastDate = today; localStorage.setItem(STREAK_KEY, JSON.stringify(streak));
  if ([3, 7, 14, 30].includes(streak.count)) { spawnConfetti(window.innerWidth / 2, window.innerHeight / 2); showToast('🔥 ' + streak.count + '-day streak! Keep it up!'); }
}

/* ═══════════════════════════════════════
   PROFILE SCREEN
═══════════════════════════════════════ */
function getProfile() { try { return JSON.parse(localStorage.getItem(PROFILE_KEY) || '{}'); } catch { return {}; } }
function renderProfile() {
  const p = getProfile();
  if (p.name) { document.getElementById('pf-name').value = p.name; document.getElementById('profile-name-disp').textContent = p.name; const avs = ['🌿', '🥦', '🍎', '🥗', '💪', '🌱', '🥑', '🍊']; document.getElementById('profile-avatar').textContent = avs[p.name.charCodeAt(0) % avs.length]; }
  if (p.age) document.getElementById('pf-age').value = p.age;
  if (p.weight) document.getElementById('pf-weight').value = p.weight;
  if (p.height) document.getElementById('pf-height').value = p.height;
  if (p.gender) document.getElementById('pf-gender').value = p.gender;
  if (p.activity) document.getElementById('pf-activity').value = p.activity;
  if (p.goal) document.querySelectorAll('#goal-row .chip').forEach(c => c.classList.toggle('active', c.dataset.goal === p.goal));
  if (p.tdee) { showTDEE(p.tdee, p.goal); updateProgressRings(p.tdee, p.goal); }
}
function computeTDEE(p) {
  if (!p.weight || !p.height || !p.age || !p.gender) return null;
  let bmr = p.gender === 'male' ? 10 * p.weight + 6.25 * p.height - 5 * p.age + 5 : 10 * p.weight + 6.25 * p.height - 5 * p.age - 161;
  const tdee = Math.round(bmr * parseFloat(p.activity || 1.55));
  return tdee + ({ lose: -500, maintain: 0, gain: 300 }[p.goal || 'maintain'] || 0);
}
function showTDEE(tdee, goal) {
  document.getElementById('tdee-card').style.display = ''; document.getElementById('macro-targets').style.display = ''; document.getElementById('today-progress').style.display = '';
  document.getElementById('tdee-val').textContent = tdee + ' kcal';
  document.getElementById('tdee-sub').textContent = { lose: 'Weight loss target', maintain: 'Maintenance target', gain: 'Muscle building target' }[goal] || 'Daily target';
  const prot = Math.round((tdee * 0.30) / 4), fat = Math.round((tdee * 0.28) / 9), carb = Math.round((tdee * 0.42) / 4);
  document.getElementById('mt-protein').textContent = prot + 'g'; document.getElementById('mt-carbs').textContent = carb + 'g'; document.getElementById('mt-fat').textContent = fat + 'g';
}
function updateProgressRings(tdee, goal) {
  const today = new Date().toDateString(), hist = getHistory().filter(h => new Date(h.ts).toDateString() === today);
  let totCal = 0, totProt = 0, totCarb = 0, totFat = 0;
  hist.forEach(h => { const n = h.data?.nutrient_breakdown || []; totCal += (n.find(x => x.name?.toLowerCase().includes('calorie'))?.value || 0); totProt += (n.find(x => x.name?.toLowerCase().includes('protein'))?.value || 0); totCarb += (n.find(x => x.name?.toLowerCase().includes('carb'))?.value || 0); totFat += (n.find(x => x.name?.toLowerCase().includes('fat'))?.value || 0); });
  const prot = Math.round((tdee * 0.30) / 4), fat = Math.round((tdee * 0.28) / 9), carb = Math.round((tdee * 0.42) / 4);
  setRing('ring-cal', totCal, tdee, totCal + '', '#FFD600'); setRing('ring-prot', totProt, prot, totProt + 'g', '#FF2D78'); setRing('ring-carb', totCarb, carb, totCarb + 'g', '#0047FF'); setRing('ring-fat', totFat, fat, totFat + 'g', '#FF6B00');
}
function setRing(id, val, max, label, color) {
  const circ = 141.4, pct = Math.min(1, val / (max || 1)), offset = circ - (pct * circ);
  const el = document.getElementById(id); if (el) el.style.strokeDashoffset = offset;
  const txt = document.getElementById(id + '-text'); if (txt) { txt.textContent = label; txt.style.color = pct > 0.9 ? 'var(--red)' : pct > 0.7 ? 'var(--orange)' : color; }
}
function saveProfile() {
  const name = document.getElementById('pf-name').value.trim(), age = parseFloat(document.getElementById('pf-age').value) || 0, weight = parseFloat(document.getElementById('pf-weight').value) || 0, height = parseFloat(document.getElementById('pf-height').value) || 0, gender = document.getElementById('pf-gender').value, activity = document.getElementById('pf-activity').value, goal = document.querySelector('#goal-row .chip.active')?.dataset.goal || 'maintain';
  const p = { name, age, weight, height, gender, activity, goal }, tdee = computeTDEE(p);
  if (tdee) p.tdee = tdee;
  localStorage.setItem(PROFILE_KEY, JSON.stringify(p));
  if (name) document.getElementById('profile-name-disp').textContent = name;
  if (tdee) { showTDEE(tdee, goal); updateProgressRings(tdee, goal); }
  spawnConfetti(window.innerWidth / 2, 200); showToast('✓ Profile saved — target: ' + (tdee || '?') + ' kcal');
}

/* ═══════════════════════════════════════
   CONSTELLATION MAP
═══════════════════════════════════════ */
function buildConstellationData(data) {
  const colorMap = { natural: '#00C896', seasoning: '#C8A800', additive: '#FF2D78', preservative: '#FF2D78', vitamin: '#C084FC', emulsifier: '#0047FF', flavour: '#FF6B00' };
  state.conIngredients = (data.ingredients_spotlight || []).map(ing => {
    const type = ing.type?.toLowerCase() || 'natural';
    return { name: ing.name, type, color: colorMap[type] || '#0047FF', size: 10 + Math.random() * 8, x: 0.08 + Math.random() * 0.84, y: 0.08 + Math.random() * 0.84, desc: ing.health_impact || ing.what_it_is || 'Ingredient in this product.', what: ing.what_it_is || '', fact: ing.curiosity_fact || '' };
  });
  if (!state.conIngredients.length) buildDemoConstellation();
}

function buildDemoConstellation() {
  state.conIngredients = [
    { name: 'Wheat flour', type: 'natural', color: '#00C896', size: 14, x: 0.18, y: 0.22, desc: 'Refined wheat, low fibre, high GI.', what: 'Refined wheat flour used as the primary structural base.', fact: 'A single wheat grain contains 22+ nutrients, almost all removed during refining.' },
    { name: 'Palm oil', type: 'natural', color: '#C8A800', size: 10, x: 0.28, y: 0.15, desc: 'Vegetable oil, high saturated fats.', what: 'Vegetable fat extracted from palm fruit.', fact: 'Palm oil is 50% saturated fat — higher than lard.' },
    { name: 'Salt', type: 'seasoning', color: '#C8A800', size: 9, x: 0.22, y: 0.38, desc: 'Sodium chloride, high sodium.', what: 'Sodium chloride used as a preservative and flavour enhancer.', fact: 'Excess sodium displaces potassium, raising blood pressure over time.' },
    { name: 'MSG (INS 621)', type: 'additive', color: '#FF2D78', size: 13, x: 0.52, y: 0.72, desc: 'Flavour enhancer, causes sensitivity in some.', what: 'Monosodium glutamate — the sodium salt of glutamic acid.', fact: 'MSG occurs naturally in tomatoes and parmesan cheese at much lower levels.' },
    { name: 'INS 211', type: 'preservative', color: '#FF2D78', size: 10, x: 0.62, y: 0.80, desc: 'Sodium benzoate, hyperactivity link in children.', what: 'A preservative that prevents mould and bacteria growth.', fact: 'INS 211 can convert to benzene (a carcinogen) in presence of vitamin C.' },
    { name: 'Tartrazine E102', type: 'additive', color: '#FF2D78', size: 11, x: 0.72, y: 0.70, desc: 'Yellow dye, banned in some countries.', what: 'A synthetic yellow azo dye used for colouring.', fact: 'Tartrazine is banned in Norway and Austria; triggers reactions in 1 in 10,000 people.' },
    { name: 'Niacin (B3)', type: 'vitamin', color: '#C084FC', size: 8, x: 0.72, y: 0.18, desc: 'Supports energy metabolism.', what: 'Water-soluble vitamin added as flour fortification.', fact: 'Niacin was added to flour in 1941 to combat pellagra, eliminating the disease in the US.' },
    { name: 'Thiamine (B1)', type: 'vitamin', color: '#C084FC', size: 7, x: 0.80, y: 0.28, desc: 'Carbohydrate metabolism vitamin.', what: 'B-vitamin essential for converting carbohydrates to energy.', fact: 'A severe thiamine deficiency causes beriberi — a disease that killed millions before rice enrichment.' },
  ];
}

let conCanvas, conCtx, conW, conH, hoveredIngr = -1;

function initConstellation() {
  conCanvas = document.getElementById('constellation-canvas'); if (!conCanvas) return;
  conW = conCanvas.offsetWidth; conH = conCanvas.offsetHeight;
  conCanvas.width = conW; conCanvas.height = conH;
  conCtx = conCanvas.getContext('2d');
  if (!state.lastResult) buildDemoConstellation();
  drawConstellation();
  conCanvas.onmousemove = e => { const r = conCanvas.getBoundingClientRect(), idx = nearestIngr(e.clientX - r.left, e.clientY - r.top); if (idx !== hoveredIngr) { hoveredIngr = idx; drawConstellation(); } };
  conCanvas.ontouchmove = e => { e.preventDefault(); const r = conCanvas.getBoundingClientRect(), t = e.touches[0], idx = nearestIngr(t.clientX - r.left, t.clientY - r.top); if (idx !== hoveredIngr) { hoveredIngr = idx; drawConstellation(); } };
  conCanvas.onclick = e => { const r = conCanvas.getBoundingClientRect(), idx = nearestIngr(e.clientX - r.left, e.clientY - r.top); if (idx >= 0) showConTooltip(idx, e.clientX, e.clientY); else hideConTooltip(); };
  conCanvas.ontouchend = e => { const r = conCanvas.getBoundingClientRect(), t = e.changedTouches[0], idx = nearestIngr(t.clientX - r.left, t.clientY - r.top); if (idx >= 0) showConTooltip(idx, t.clientX, t.clientY); };
}
function ix(i) { return state.conIngredients[i].x * conW; }
function iy(i) { return state.conIngredients[i].y * conH; }
function nearestIngr(px, py) {
  let best = -1, bestD = 999;
  state.conIngredients.forEach((ing, i) => { const dx = ix(i) - px, dy = iy(i) - py, d = Math.sqrt(dx * dx + dy * dy); if (d < ing.size * 2.8 && d < bestD) { bestD = d; best = i; } });
  return best;
}
function drawConstellation() {
  if (!conCtx) return;
  conCtx.clearRect(0, 0, conW, conH); conCtx.fillStyle = 'rgba(245,243,238,0.4)'; conCtx.fillRect(0, 0, conW, conH);
  [{ x: 0.2, y: 0.25, r: 0.15, c: 'rgba(0,200,150,0.06)' }, { x: 0.75, y: 0.23, r: 0.12, c: 'rgba(192,132,252,0.06)' }, { x: 0.55, y: 0.76, r: 0.17, c: 'rgba(255,45,120,0.06)' }, { x: 0.48, y: 0.48, r: 0.11, c: 'rgba(200,168,0,0.05)' }].forEach(n => {
    const g = conCtx.createRadialGradient(n.x * conW, n.y * conH, 0, n.x * conW, n.y * conH, n.r * Math.min(conW, conH));
    g.addColorStop(0, n.c); g.addColorStop(1, 'transparent');
    conCtx.beginPath(); conCtx.arc(n.x * conW, n.y * conH, n.r * Math.min(conW, conH), 0, Math.PI * 2); conCtx.fillStyle = g; conCtx.fill();
  });
  state.conIngredients.forEach((a, i) => { state.conIngredients.forEach((b, j) => { if (j <= i || a.type !== b.type) return; const dx = ix(i) - ix(j), dy = iy(i) - iy(j); if (Math.sqrt(dx * dx + dy * dy) > conW * 0.28) return; conCtx.beginPath(); conCtx.moveTo(ix(i), iy(i)); conCtx.lineTo(ix(j), iy(j)); conCtx.strokeStyle = a.color + '22'; conCtx.lineWidth = 1; conCtx.stroke(); }); });
  state.conIngredients.forEach((ing, i) => {
    const x = ix(i), y = iy(i), r = ing.size * (hoveredIngr === i ? 1.45 : 1), hov = hoveredIngr === i;
    const grd = conCtx.createRadialGradient(x, y, 0, x, y, r * 3.5); grd.addColorStop(0, ing.color + (hov ? '28' : '14')); grd.addColorStop(1, 'transparent');
    conCtx.beginPath(); conCtx.arc(x, y, r * 3.5, 0, Math.PI * 2); conCtx.fillStyle = grd; conCtx.fill();
    conCtx.beginPath(); conCtx.arc(x, y, r, 0, Math.PI * 2); conCtx.fillStyle = ing.color + (hov ? 'EE' : 'BB'); conCtx.fill();
    conCtx.lineWidth = 1.5; conCtx.strokeStyle = 'rgba(10,10,10,0.4)'; conCtx.stroke();
    if (hov) { conCtx.beginPath(); conCtx.arc(x, y, r + 8, 0, Math.PI * 2); conCtx.strokeStyle = ing.color + '55'; conCtx.lineWidth = 1.5; conCtx.stroke(); conCtx.font = `700 11px 'Nunito',sans-serif`; conCtx.fillStyle = 'rgba(10,10,10,0.85)'; conCtx.textAlign = x < conW * 0.6 ? 'left' : 'right'; conCtx.fillText(ing.name, x + (x < conW * 0.6 ? r + 12 : -r - 12), y + 4); }
  });
}
function showConTooltip(idx, cx, cy) {
  const ing = state.conIngredients[idx], tt = document.getElementById('ingr-tooltip');
  document.getElementById('tt-name').textContent = ing.name;
  const types = { natural: 'Natural', seasoning: 'Seasoning', additive: 'Additive', preservative: 'Preservative', vitamin: 'Vitamin', emulsifier: 'Emulsifier', flavour: 'Flavour' };
  const ttType = document.getElementById('tt-type'); ttType.textContent = types[ing.type] || ing.type; ttType.style.color = ing.color;
  document.getElementById('tt-desc').textContent = ing.desc;
  let left = cx + 14, top = cy - 70; if (left + 215 > window.innerWidth) left = cx - 225; if (top < 10) top = cy + 14;
  tt.style.left = left + 'px'; tt.style.top = top + 'px'; tt.classList.add('show');
}
function hideConTooltip() { document.getElementById('ingr-tooltip').classList.remove('show'); }

/* ═══════════════════════════════════════
   HISTORY (localStorage)
═══════════════════════════════════════ */
function getHistory() { try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch { return []; } }
function saveHistory(result, thumb) {
  const hist = getHistory();
  hist.unshift({ id: Date.now(), ts: new Date().toISOString(), product_name: result.product_name || 'Unknown', score: result.score || 0, verdict: result.verdict || '', category: result.product_category || '', language: state.language, thumb: thumb || null, is_voice: result.is_voice_log || false, is_barcode: result.is_barcode_scan || false, data: result });
  if (hist.length > 50) hist.pop();
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(hist)); } catch { }
}
function clearHistory() {
  if (!confirm('Clear all history?')) return;
  localStorage.removeItem(HISTORY_KEY); localStorage.removeItem(STREAK_KEY);
  renderHistory(); showToast('🗑 History cleared');
}

/* ═══════════════════════════════════════
   QUOTA + PAYWALL
═══════════════════════════════════════ */
async function loadQuota() {
  try { const res = await fetch(`${API}/scan-quota`), d = await res.json(); state.quotaRemaining = d.remaining ?? d.scans_remaining ?? 5; state.isPro = d.is_pro ?? false; updateQuota(); } catch { }
}
function updateQuota() {
  const t = document.getElementById('quota-text'), dot = document.getElementById('qdot');
  if (state.isPro) { t.textContent = '∞'; dot.className = 'quota-dot'; }
  else { t.textContent = state.quotaRemaining + ' left'; dot.className = state.quotaRemaining <= 1 ? 'quota-dot low' : state.quotaRemaining <= 3 ? 'quota-dot warn' : 'quota-dot'; }
}
function openPaywall(trigger = '') {
  if (trigger === 'limit') document.getElementById('pw-sub').textContent = "You've used all free scans today. Upgrade to keep going.";
  document.getElementById('paywall-overlay').classList.add('open');
}
function closePaywall() { document.getElementById('paywall-overlay').classList.remove('open'); }
async function activatePro() {
  try {
    const fd = new FormData(); fd.append('payment_id', 'demo_' + Date.now());
    const res = await fetch(`${API}/activate-pro`, { method: 'POST', body: fd }), d = await res.json();
    if (d.status === 'activated') { state.isPro = true; state.quotaRemaining = 99999; closePaywall(); updateQuota(); showToast('🎉 Pro activated!'); spawnConfetti(window.innerWidth / 2, window.innerHeight / 2); }
  } catch { showToast('❌ Payment failed.'); }
}

/* ═══════════════════════════════════════
   SHARE + PORTAL RESET
═══════════════════════════════════════ */
async function shareResult() {
  if (!state.lastResult) { showToast('⚠️ No result to share'); return; }
  const d = state.lastResult, text = `Eatlytic — ${d.product_name} — ${d.score}/10\n${d.verdict}\n\n${d.summary}`;
  try { if (navigator.share) await navigator.share({ title: 'Eatlytic', text }); else { await navigator.clipboard.writeText(text); showToast('📋 Copied!'); } } catch { showToast('📋 Share unavailable'); }
}
function resetPortal() {
  state.blob = null;
  const img = document.getElementById('preview-img'); if (img.src) URL.revokeObjectURL(img.src); img.src = '';
  document.getElementById('portal-preview-wrap').classList.remove('visible');
  document.getElementById('analyse-cta').classList.remove('visible');
  document.getElementById('quality-strip').classList.remove('visible');
}

/* ═══════════════════════════════════════
   CONFETTI + STICKER POP + TOAST
═══════════════════════════════════════ */
function spawnConfetti(cx, cy) {
  for (let i = 0; i < 22; i++) {
    const p = document.createElement('div'); p.className = 'confetti-piece';
    p.style.cssText = `left:${cx + (Math.random() - .5) * 80}px;top:${cy - 8}px;background:${CONFETTI_COLORS[i % CONFETTI_COLORS.length]};transform:rotate(${Math.random() * 360}deg);animation-duration:${0.7 + Math.random() * .6}s;animation-delay:${Math.random() * .18}s;border-radius:${Math.random() > .5 ? '50%' : '2px'}`;
    document.body.appendChild(p); setTimeout(() => p.remove(), 1500);
  }
}
function spawnStickerPop(emoji, cx, cy) {
  const p = document.createElement('div'); p.className = 'sticker-pop'; p.textContent = emoji;
  p.style.cssText = `left:${cx - 20}px;top:${cy - 30}px`; document.body.appendChild(p); setTimeout(() => p.remove(), 1000);
}
let toastTimer;
function showToast(msg) {
  const t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 2800);
}

function esc(s) { if (!s) return ''; return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }

/* --- FEEDBACK LOOP logic --- */
async function reportScanError(scanId) {
  const note = prompt("What's wrong? (e.g., 'Wrong protein amount', 'Didn't detect sugar')");
  if (note === null) return;
  try {
    const fd = new FormData(); fd.append('scan_id', scanId); fd.append('note', note);
    const res = await fetch(`${API}/api/v1/report-error`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error();
    showToast('🚩 Reported. We will review this scan!');
    document.querySelector('.report-error-btn').style.display = 'none';
  } catch {
    showToast('❌ Error reporting failed.');
  }
}

/* ═══════════════════════════════════════
   INIT
═══════════════════════════════════════ */
document.getElementById('bottom-nav').classList.add('visible');
loadQuota();
setupVoice();
setTimeout(() => { const p = getProfile(); if (p.name) document.getElementById('profile-name-disp').textContent = p.name; }, 100);
window.addEventListener('popstate', () => { if (state.cameraStream) stopCamera(); });

/* ═══════════════════════════════════════
   DARK MODE
═══════════════════════════════════════ */
function toggleDark() { setDark(document.documentElement.getAttribute('data-theme') !== 'dark'); }
function setDark(on) {
  document.documentElement.setAttribute('data-theme', on ? 'dark' : 'light');
  const btn = document.getElementById('theme-toggle'); if (btn) btn.textContent = on ? '☀️' : '🌙';
  try { localStorage.setItem('eatlytic-theme', on ? 'dark' : 'light'); } catch { }
}
(function initTheme() {
  let saved = null; try { saved = localStorage.getItem('eatlytic-theme'); } catch { }
  if (saved) setDark(saved === 'dark');
  else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) setDark(true);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => { let s = null; try { s = localStorage.getItem('eatlytic-theme'); } catch { } if (!s) setDark(e.matches); });
})();

/* ═══════════════════════════════════════
   KEYBOARD / NAV FIX FOR MOBILE
═══════════════════════════════════════ */
(function initNavKeyboardFix() {
  const nav = document.getElementById('bottom-nav'); if (!nav) return;
  function onViewportResize() {
    const keyboardOpen = window.visualViewport ? window.visualViewport.height < window.innerHeight * 0.75 : false;
    nav.style.transform = keyboardOpen ? 'translateY(110%)' : ''; nav.style.transition = 'transform 0.2s ease';
  }
  if (window.visualViewport) { window.visualViewport.addEventListener('resize', onViewportResize); window.visualViewport.addEventListener('scroll', onViewportResize); }
  document.addEventListener('focusin', e => { if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) { nav.style.transform = 'translateY(110%)'; nav.style.transition = 'transform 0.2s ease'; } });
  document.addEventListener('focusout', () => { setTimeout(() => { const ko = window.visualViewport ? window.visualViewport.height < window.innerHeight * 0.75 : false; if (!ko) nav.style.transform = ''; }, 150); });
})();

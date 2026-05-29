import re
import secrets
import json
import hashlib
from fastapi import APIRouter, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse
from app.database.connection import db_conn

router = APIRouter(prefix="/dietitian", tags=["dietitian"])

@router.post("/register")
async def register_dietitian(
    name: str = Form(...),
    email: str = Form(...),
    code: str = Form(...)
):
    """
    Registers a clinical dietitian/nutritionist and provides them with a cohort 
    code and dietitian key to access aggregate patient insights.
    """
    cleaned_code = re.sub(r"[^\w]", "", code.strip().upper())
    if not cleaned_code:
        raise HTTPException(status_code=400, detail="Invalid cohort code.")
    
    # Generate secure admin access key
    key = "diet_" + secrets.token_urlsafe(32)
    
    with db_conn() as conn:
        existing = conn.execute("SELECT code FROM dietitians WHERE code=?", (cleaned_code,)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Cohort code is already registered.")
        
        conn.execute(
            "INSERT INTO dietitians (code, name, email, dietitian_key) VALUES (?, ?, ?, ?)",
            (cleaned_code, name.strip(), email.strip().lower(), key)
        )
        
    return {
        "status": "registered",
        "cohort_code": cleaned_code,
        "dietitian_key": key,
        "dashboard_url": f"/dietitian/dashboard?key={key}"
    }

@router.post("/join")
async def join_cohort(
    device_key: str = Form(...),
    cohort_code: str = Form(...)
):
    """
    Ties a patient's device_key to a dietitian's tracking cohort code.
    All subsequent scans are aggregated in the dietitian dashboard.
    """
    cleaned_code = cohort_code.strip().upper()
    with db_conn() as conn:
        dietitian = conn.execute("SELECT code, name FROM dietitians WHERE code=?", (cleaned_code,)).fetchone()
        if not dietitian:
            raise HTTPException(status_code=404, detail="Cohort code not found.")
        
        conn.execute(
            "INSERT OR IGNORE INTO patient_cohorts (device_key, dietitian_code) VALUES (?, ?)",
            (device_key.strip(), cleaned_code)
        )
        
    return {
        "status": "joined",
        "cohort_code": cleaned_code,
        "dietitian_name": dietitian["name"],
        "message": f"Successfully joined {dietitian['name']}'s cohort!"
    }

DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Eatlytic Clinical Console | Dietitian Dashboard</title>
  
  <!-- Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,700;0,9..144,900;1,9..144,700&family=Nunito:wght@400;600;700;800;900&display=swap" rel="stylesheet">
  
  <style>
    :root {
      --bg: #F5F3EE;
      --ink: #0A0A0A;
      --white: #FFFFFF;
      --yellow: #FFD600;
      --pink: #FF2D78;
      --mint: #00C896;
      --blue: #0047FF;
      --orange: #FF6B00;
      --lilac: #C084FC;
      --cream: #FFF8E8;
      --border: 2.5px solid var(--ink);
      --shadow: 4px 4px 0px var(--ink);
      --shadow-lg: 8px 8px 0px var(--ink);
      --shadow-sm: 2px 2px 0px var(--ink);
    }
    [data-theme="dark"] {
      --bg: #111111;
      --ink: #F0EDE6;
      --white: #1A1A1A;
      --yellow: #E6C000;
      --pink: #E11D48;
      --mint: #00A37A;
      --blue: #2563EB;
      --cream: #1E1C18;
      --border: 2.5px solid rgba(240, 237, 230, 0.2);
      --shadow: 4px 4px 0px rgba(240, 237, 230, 0.15);
      --shadow-lg: 8px 8px 0px rgba(240, 237, 230, 0.15);
      --shadow-sm: 2px 2px 0px rgba(240, 237, 230, 0.15);
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    body {
      background-color: var(--bg);
      color: var(--ink);
      font-family: 'Nunito', sans-serif;
      line-height: 1.6;
      padding: 0;
      margin: 0;
    }
    body::after {
      content: '';
      position: fixed;
      inset: 0;
      z-index: -1;
      pointer-events: none;
      background-image: radial-gradient(circle, rgba(10, 10, 10, 0.04) 1px, transparent 1px);
      background-size: 16px 16px;
    }
    [data-theme="dark"] body::after {
      background-image: radial-gradient(circle, rgba(240, 237, 230, 0.03) 1px, transparent 1px);
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 30px 20px;
    }
    header.app-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: var(--border);
      padding-bottom: 20px;
      margin-bottom: 30px;
    }
    .header-left {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .logo {
      font-family: 'Fraunces', serif;
      font-weight: 900;
      font-size: 2rem;
      font-style: italic;
      text-decoration: none;
      color: var(--ink);
    }
    .logo em {
      color: var(--yellow);
      -webkit-text-stroke: 1px var(--ink);
    }
    .clinical-badge {
      background: var(--pink);
      border: var(--border);
      color: var(--white);
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      font-size: 11px;
      padding: 4px 10px;
      border-radius: 4px;
      box-shadow: var(--shadow-sm);
    }
    .btn {
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      text-decoration: none;
      color: var(--ink);
      border: var(--border);
      padding: 8px 18px;
      border-radius: 8px;
      background: var(--white);
      font-size: 13px;
      box-shadow: var(--shadow-sm);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: transform 0.1s, box-shadow 0.1s;
    }
    .btn:hover {
      transform: translate(-2px, -2px);
      box-shadow: 4px 4px 0px var(--ink);
    }
    .btn:active {
      transform: translate(2px, 2px);
      box-shadow: none;
    }
    .btn-primary {
      background: var(--yellow);
    }
    .btn-secondary {
      background: var(--lilac);
    }
    .btn-action {
      background: var(--mint);
      font-size: 11px;
      padding: 6px 12px;
    }
    .dashboard-hero {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 40px;
      gap: 20px;
      flex-wrap: wrap;
    }
    .hero-text h1 {
      font-family: 'Fraunces', serif;
      font-weight: 900;
      font-size: 2.2rem;
      margin-bottom: 6px;
    }
    .highlight-text {
      color: var(--blue);
      text-decoration: underline;
      text-decoration-color: var(--yellow);
      text-decoration-thickness: 4px;
    }
    [data-theme="dark"] .highlight-text {
      color: var(--lilac);
    }
    .hero-text p {
      color: var(--ink);
      opacity: 0.8;
      font-size: 15px;
    }
    .cohort-sticker {
      background: var(--cream);
      border: var(--border);
      border-radius: 12px;
      padding: 16px 20px;
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .sticker-label {
      font-family: 'Space Mono', monospace;
      font-size: 10px;
      font-weight: 700;
      color: var(--ink);
      opacity: 0.6;
    }
    .sticker-code-wrapper {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .sticker-code-wrapper code {
      font-family: 'Space Mono', monospace;
      font-size: 22px;
      font-weight: 700;
      color: var(--ink);
    }
    .copy-btn {
      background: transparent;
      border: none;
      font-size: 18px;
      cursor: pointer;
      transition: transform 0.1s;
    }
    .copy-btn:hover {
      transform: scale(1.2);
    }
    .copy-btn:active {
      transform: scale(0.9);
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 24px;
      margin-bottom: 40px;
    }
    .card {
      background: var(--white);
      border: var(--border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }
    .stat-card {
      display: flex;
      align-items: center;
      gap: 20px;
    }
    .stat-icon {
      font-size: 32px;
      width: 60px;
      height: 60px;
      border-radius: 12px;
      border: var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: var(--shadow-sm);
    }
    .stat-icon.patients { background: var(--lilac); }
    .stat-icon.scans { background: var(--yellow); }
    .stat-icon.alerts { background: var(--pink); }
    .stat-info {
      display: flex;
      flex-direction: column;
    }
    .stat-num {
      font-family: 'Space Mono', monospace;
      font-size: 28px;
      font-weight: 700;
      line-height: 1;
      margin-bottom: 4px;
    }
    .stat-lbl {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--ink);
      opacity: 0.6;
    }
    .dashboard-grid {
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 30px;
      margin-bottom: 40px;
    }
    @media (max-width: 992px) {
      .dashboard-grid {
        grid-template-columns: 1fr;
      }
    }
    .section-title {
      font-family: 'Fraunces', serif;
      font-weight: 900;
      font-size: 1.6rem;
      margin-bottom: 8px;
    }
    .section-desc {
      font-size: 13px;
      color: var(--ink);
      opacity: 0.7;
      margin-bottom: 24px;
      line-height: 1.4;
    }
    .distribution-bar {
      display: flex;
      height: 28px;
      border: var(--border);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 12px;
      box-shadow: var(--shadow-sm);
    }
    .bar-segment {
      height: 100%;
      transition: width 0.5s ease-in-out;
    }
    .safe-segment { background: var(--mint); }
    .limit-segment { background: var(--yellow); }
    .avoid-segment { background: var(--pink); }
    .distribution-legend {
      display: flex;
      justify-content: space-between;
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 10px;
    }
    .legend-item {
      font-family: 'Space Mono', monospace;
      font-size: 11px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .color-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      border: 1.5px solid var(--ink);
      display: inline-block;
    }
    .safe-dot { background: var(--mint); }
    .limit-dot { background: var(--yellow); }
    .avoid-dot { background: var(--pink); }
    .distribution-stickers-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 12px;
    }
    .dist-sticker {
      border: var(--border);
      border-radius: 12px;
      padding: 14px 10px;
      text-align: center;
      box-shadow: var(--shadow-sm);
    }
    .safe-sticker { background: var(--mint); }
    .limit-sticker { background: var(--yellow); }
    .avoid-sticker { background: var(--pink); }
    .dist-val {
      font-family: 'Space Mono', monospace;
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .dist-lbl {
      font-size: 8px;
      font-weight: 800;
      letter-spacing: 0.5px;
    }
    .threat-list {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .threat-item-card {
      border: var(--border);
      border-radius: 12px;
      background: var(--cream);
      padding: 20px;
      box-shadow: var(--shadow-sm);
      transition: transform 0.15s, box-shadow 0.15s;
    }
    .threat-item-card:hover {
      transform: translate(-2px, -2px);
      box-shadow: 4px 4px 0px var(--ink);
    }
    .threat-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      border-bottom: 1.5px dashed rgba(10, 10, 10, 0.15);
      padding-bottom: 8px;
    }
    [data-theme="dark"] .threat-header {
      border-bottom: 1.5px dashed rgba(240, 237, 230, 0.15);
    }
    .threat-patient-id {
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 6px;
      background: var(--white);
      padding: 2px 8px;
      border-radius: 4px;
      border: var(--border);
    }
    .threat-date {
      font-family: 'Space Mono', monospace;
      font-size: 11px;
      color: var(--ink);
      opacity: 0.6;
    }
    .threat-product {
      font-family: 'Fraunces', serif;
      font-weight: 900;
      font-size: 1.3rem;
      margin-bottom: 10px;
    }
    .threat-brand {
      font-family: 'Nunito', sans-serif;
      font-size: 12px;
      font-weight: 700;
      background: var(--white);
      border: var(--border);
      padding: 1px 6px;
      border-radius: 4px;
      vertical-align: middle;
      margin-left: 6px;
      opacity: 0.9;
    }
    .threat-metrics-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }
    .threat-metric {
      background: var(--white);
      border: var(--border);
      border-radius: 8px;
      padding: 8px;
      text-align: center;
    }
    .metric-lbl {
      font-size: 9px;
      font-weight: 800;
      text-transform: uppercase;
      color: var(--ink);
      opacity: 0.5;
      display: block;
    }
    .metric-val {
      font-family: 'Space Mono', monospace;
      font-size: 14px;
      font-weight: 700;
      display: block;
      margin-top: 2px;
    }
    .metric-val.alert-danger {
      color: var(--pink);
    }
    .threat-triggers-section {
      background: var(--white);
      border: var(--border);
      border-radius: 8px;
      padding: 12px;
    }
    .trigger-label {
      font-family: 'Space Mono', monospace;
      font-size: 9px;
      font-weight: 800;
      color: var(--pink);
      margin-bottom: 6px;
      letter-spacing: 0.5px;
    }
    .trigger-list {
      padding-left: 18px;
      font-size: 12px;
      line-height: 1.4;
      margin: 0;
    }
    .trigger-list li {
      margin-bottom: 4px;
    }
    .threat-footer {
      display: flex;
      justify-content: flex-end;
      margin-top: 14px;
    }
    .table-responsive {
      width: 100%;
      overflow-x: auto;
      border: var(--border);
      border-radius: 12px;
      box-shadow: var(--shadow-sm);
      background: var(--white);
    }
    .neobrutalist-table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 13px;
    }
    .neobrutalist-table th,
    .neobrutalist-table td {
      padding: 12px 16px;
      border-bottom: var(--border);
      border-right: var(--border);
    }
    .neobrutalist-table th:last-child,
    .neobrutalist-table td:last-child {
      border-right: none;
    }
    .neobrutalist-table tr:last-child td {
      border-bottom: none;
    }
    .neobrutalist-table th {
      background: var(--cream);
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      text-transform: uppercase;
      font-size: 11px;
    }
    .neobrutalist-table tbody tr:hover {
      background: var(--bg);
    }
    .product-cell {
      font-weight: 700;
    }
    .score-badge {
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      border: var(--border);
      padding: 2px 8px;
      border-radius: 4px;
      box-shadow: 1px 1px 0px var(--ink);
    }
    .score-badge.badge-safe { background: var(--mint); }
    .score-badge.badge-limit { background: var(--yellow); }
    .score-badge.badge-avoid { background: var(--pink); }
    .tier-pill {
      font-family: 'Space Mono', monospace;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      padding: 2px 6px;
      border-radius: 100px;
      border: 1.5px solid var(--ink);
    }
    .tier-pill.tier-safe { background: var(--mint); }
    .tier-pill.tier-limit { background: var(--yellow); }
    .tier-pill.tier-avoid { background: var(--pink); }
    .time-cell {
      font-family: 'Space Mono', monospace;
      opacity: 0.7;
    }
    .empty-state-card {
      text-align: center;
      padding: 40px 20px;
      background: var(--cream);
    }
    .empty-icon {
      font-size: 48px;
      margin-bottom: 16px;
    }
    .empty-state-card h3 {
      font-family: 'Fraunces', serif;
      font-size: 1.4rem;
      margin-bottom: 8px;
    }
    .empty-state-card p {
      font-size: 13px;
      opacity: 0.8;
      max-width: 400px;
      margin: 0 auto;
    }
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(10, 10, 10, 0.6);
      backdrop-filter: blur(4px);
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: center;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease-in-out;
    }
    .modal-backdrop.show {
      opacity: 1;
      pointer-events: all;
    }
    .modal {
      background: var(--white);
      border: var(--border);
      border-radius: 16px;
      width: 90%;
      max-width: 550px;
      padding: 30px;
      box-shadow: var(--shadow-lg);
      transform: scale(0.9) translateY(20px);
      transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .modal-backdrop.show .modal {
      transform: scale(1) translateY(0);
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      border-bottom: var(--border);
      padding-bottom: 12px;
    }
    .modal-title {
      font-family: 'Fraunces', serif;
      font-weight: 900;
      font-size: 1.4rem;
    }
    .close-modal-btn {
      background: transparent;
      border: none;
      font-size: 24px;
      cursor: pointer;
      line-height: 1;
    }
    .modal-body {
      margin-bottom: 24px;
    }
    .textarea-intervention {
      width: 100%;
      height: 160px;
      border: var(--border);
      border-radius: 8px;
      padding: 12px;
      font-family: inherit;
      font-size: 13px;
      line-height: 1.5;
      background: var(--bg);
      color: var(--ink);
      resize: none;
    }
    .modal-footer {
      display: flex;
      justify-content: flex-end;
      gap: 12px;
    }
    .toast {
      position: fixed;
      bottom: 30px;
      left: 50%;
      transform: translateX(-50%) translateY(100px);
      background: var(--ink);
      color: var(--bg);
      border: var(--border);
      box-shadow: var(--shadow);
      padding: 12px 24px;
      border-radius: 8px;
      font-weight: 800;
      z-index: 1010;
      transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .toast.show {
      transform: translateX(-50%) translateY(0);
    }
  </style>
</head>
<body>
  <div class="container">
    <header class="app-header">
      <div class="header-left">
        <a href="#" class="logo">Eat<em>l</em>ytic</a>
        <span class="badge clinical-badge">CLINICAL CONSOLE</span>
      </div>
      <div class="nav-actions">
        <button id="theme-toggle" class="btn" onclick="toggleTheme()">🌙 Mode</button>
        <a href="/static/developer.html" class="btn btn-secondary">Developer Portal</a>
      </div>
    </header>

    <main>
      <!-- Hero Dashboard Section -->
      <div class="dashboard-hero">
        <div class="hero-text">
          <h1>Welcome back, <span class="highlight-text">Dr. {{dietitian_name}}</span></h1>
          <p>Clinical console for patient cohort analytics and real-time metabolic threat tracking.</p>
        </div>
        <div class="cohort-sticker">
          <div class="sticker-label">ACTIVE COHORT CODE</div>
          <div class="sticker-code-wrapper">
            <code id="cohort-code-val">{{cohort_code}}</code>
            <button class="copy-btn" onclick="copyCohortCode()" title="Copy Code">📋</button>
          </div>
        </div>
      </div>

      <!-- Stats Grid -->
      <section class="stats-grid">
        <div class="card stat-card">
          <div class="stat-icon patients">👥</div>
          <div class="stat-info">
            <span class="stat-num">{{total_patients}}</span>
            <span class="stat-lbl">Active Patients</span>
          </div>
        </div>
        <div class="card stat-card">
          <div class="stat-icon scans">📊</div>
          <div class="stat-info">
            <span class="stat-num">{{total_scans_30d}}</span>
            <span class="stat-lbl">30d Total Scans</span>
          </div>
        </div>
        <div class="card stat-card">
          <div class="stat-icon alerts">🔴</div>
          <div class="stat-info">
            <span class="stat-num">{{threat_feed_count}}</span>
            <span class="stat-lbl">Critical Alerts</span>
          </div>
        </div>
      </section>

      <!-- Main Layout Grid -->
      <section class="dashboard-grid">
        <!-- Left Side Panel (Safety distribution) -->
        <div>
          <div class="card">
            <h2 class="section-title">Cohort Safety Breakdown</h2>
            <p class="section-desc">Aggregate nutrition quality of all scans mapped over the last 30 days.</p>
            
            <div class="distribution-bar">
              <div class="bar-segment safe-segment" style="width: {{safe_ratio}}%" title="Safe: {{safe_ratio}}%"></div>
              <div class="bar-segment limit-segment" style="width: {{limit_ratio}}%" title="Limit: {{limit_ratio}}%"></div>
              <div class="bar-segment avoid-segment" style="width: {{avoid_ratio}}%" title="Avoid: {{avoid_ratio}}%"></div>
            </div>
            
            <div class="distribution-legend">
              <div class="legend-item"><span class="color-dot safe-dot"></span> Safe ({{safe_ratio}}%)</div>
              <div class="legend-item"><span class="color-dot limit-dot"></span> Limit ({{limit_ratio}}%)</div>
              <div class="legend-item"><span class="color-dot avoid-dot"></span> Avoid ({{avoid_ratio}}%)</div>
            </div>
            
            <div class="distribution-stickers-grid">
              <div class="dist-sticker safe-sticker">
                <div class="dist-val">{{safe_ratio}}%</div>
                <div class="dist-lbl">SAFE</div>
              </div>
              <div class="dist-sticker limit-sticker">
                <div class="dist-val">{{limit_ratio}}%</div>
                <div class="dist-lbl">LIMIT</div>
              </div>
              <div class="dist-sticker avoid-sticker">
                <div class="dist-val">{{avoid_ratio}}%</div>
                <div class="dist-lbl">AVOID</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Right Side Panel (Alert Feed) -->
        <div>
          <div class="card">
            <h2 class="section-title">🔴 Glycemic & Safety Alerts</h2>
            <p class="section-desc">Real-time alerts of clinical risk profiles (high sugar/sodium, maida, trans-fats) scanned by patients.</p>
            
            <div class="threat-list">
              {{threat_html}}
            </div>
          </div>
        </div>
      </section>

      <!-- Recent Scans Timeline Table -->
      <section class="card recent-activity-card">
        <h2 class="section-title">📋 Recent Activity Timeline</h2>
        <p class="section-desc">Chronological flow of all product scans executed by patients connected to your cohort.</p>
        
        <div class="table-responsive">
          <table class="neobrutalist-table">
            <thead>
              <tr>
                <th>Patient ID</th>
                <th>Product Name</th>
                <th>Brand</th>
                <th>Score</th>
                <th>Verdict</th>
                <th>Tier</th>
                <th>Time (UTC)</th>
              </tr>
            </thead>
            <tbody>
              {{scans_html}}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  </div>

  <!-- Outreach Modal -->
  <div id="intervention-modal-backdrop" class="modal-backdrop" onclick="closeInterventionModal(event)">
    <div class="modal" onclick="event.stopPropagation()">
      <div class="modal-header">
        <h3 class="modal-title">🩺 Draft Clinical Intervention</h3>
        <button class="close-modal-btn" onclick="hideInterventionModal()">×</button>
      </div>
      <div class="modal-body">
        <p style="font-size: 11px; font-weight: 800; margin-bottom: 8px; text-transform: uppercase; color: var(--pink); letter-spacing: 0.5px;">DPDP COMPLIANT SMS/WHATSAPP OUTREACH TEXT:</p>
        <textarea id="intervention-text" class="textarea-intervention"></textarea>
      </div>
      <div class="modal-footer">
        <button class="btn btn-secondary" onclick="hideInterventionModal()">Cancel</button>
        <button class="btn btn-primary" onclick="copyInterventionText()">📋 Copy outreach Text</button>
      </div>
    </div>
  </div>

  <!-- Toast -->
  <div id="toast" class="toast">✓ Copied to clipboard</div>

  <script>
    function toggleTheme() {
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      document.documentElement.setAttribute('data-theme', isDark ? 'light' : 'dark');
      document.getElementById('theme-toggle').textContent = isDark ? '🌙 Mode' : '☀️ Mode';
      localStorage.setItem('theme', isDark ? 'light' : 'dark');
    }

    // Restore theme from localStorage on load
    window.addEventListener('DOMContentLoaded', () => {
      const storedTheme = localStorage.getItem('theme');
      if (storedTheme) {
        document.documentElement.setAttribute('data-theme', storedTheme);
        const btn = document.getElementById('theme-toggle');
        if (btn) {
          btn.textContent = storedTheme === 'dark' ? '☀️ Mode' : '🌙 Mode';
        }
      }
    });

    function copyCohortCode() {
      const code = document.getElementById('cohort-code-val').textContent;
      navigator.clipboard.writeText(code).then(() => {
        showToast('✓ Cohort code copied to clipboard!');
      }).catch(() => {
        showToast('❌ Copy failed');
      });
    }

    function showInterventionModal(patientId, productName, verdict) {
      const clinicianName = "{{dietitian_name}}";
      const text = `Hi, this is ${clinicianName} here. I noticed you recently scanned ${productName} (Verdict: ${verdict}). Given your metabolic profile and insulin sensitivity constraints, this product carries refined flour/added sugar risks that can trigger blood sugar spikes. Please review the healthy alternatives I\\'ve approved for you in your Eatlytic app!`;
      document.getElementById('intervention-text').value = text;
      document.getElementById('intervention-modal-backdrop').classList.add('show');
    }

    function hideInterventionModal() {
      document.getElementById('intervention-modal-backdrop').classList.remove('show');
    }

    function closeInterventionModal(e) {
      if (e.target.id === 'intervention-modal-backdrop') {
        hideInterventionModal();
      }
    }

    function copyInterventionText() {
      const textarea = document.getElementById('intervention-text');
      textarea.select();
      navigator.clipboard.writeText(textarea.value).then(() => {
        showToast('✓ Intervention outreach copied!');
        hideInterventionModal();
      }).catch(() => {
        showToast('❌ Copy failed');
      });
    }

    let toastTimer;
    function showToast(msg) {
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.classList.add('show');
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => t.classList.remove('show'), 2800);
    }
  </script>
</body>
</html>"""

@router.get("/dashboard")
async def get_dietitian_dashboard(
    request: Request,
    key: str = Query(...),
    format: str = Query(None)
):
    """
    Provides clinical nutritionist with aggregate cohort stats, patient-by-patient 
    telemetry, and the Glycemic Threat Feed for patient intervention.
    """
    with db_conn() as conn:
        dietitian = conn.execute("SELECT * FROM dietitians WHERE dietitian_key=?", (key.strip(),)).fetchone()
        if not dietitian:
            raise HTTPException(status_code=403, detail="Invalid dietitian key.")
        
        code = dietitian["code"]
        
        # Fetch connected device keys
        cohort_rows = conn.execute("SELECT device_key FROM patient_cohorts WHERE dietitian_code=?", (code,)).fetchall()
        device_keys = [r["device_key"] for r in cohort_rows]
        
        if not device_keys:
            total_scans = 0
            scan_rows = []
        else:
            # Compile dynamically parameterized SQL
            placeholders = ",".join("?" for _ in device_keys)
            scans_query = f"""
                SELECT id, device_key, product_name, score, verdict, persona, scanned_at, brand, category, analysis_json
                FROM scans
                WHERE device_key IN ({placeholders}) AND scanned_at >= datetime('now', '-30 days')
                ORDER BY scanned_at DESC
            """
            scan_rows = conn.execute(scans_query, device_keys).fetchall()
            total_scans = len(scan_rows)
        
    # Safety distributions
    safe_count = 0
    limit_count = 0
    avoid_count = 0
    
    glycemic_threat_feed = []
    recent_scans = []
    
    for r in scan_rows:
        row_dict = dict(r)
        score = row_dict["score"]
        
        if score >= 7:
            safe_count += 1
            tier = "Safe"
        elif score >= 4:
            limit_count += 1
            tier = "Limit"
        else:
            avoid_count += 1
            tier = "Avoid"
            
        try:
            analysis = json.loads(row_dict["analysis_json"])
        except:
            analysis = {}
            
        # Determine if this triggers a Glycemic or Health threat for diabetic patients
        is_glycemic_threat = (
            score < 4 or
            row_dict["verdict"] == "Glycemic Threat" or
            tier == "Avoid" or
            any(x in row_dict["product_name"].lower() for x in ["maltodextrin", "maida", "jim jam"]) or
            any("glycemic" in str(c).lower() for c in analysis.get("cons", [])) or
            any(any(kw in str(c).lower() for kw in ["high in added sugar", "excessive sugar", "glycemic threat", "insulin spike"]) for c in analysis.get("cons", []))
        )
        
        # Short MD5 hash of device key for anonymized Patient ID
        patient_id = f"PATIENT_{hashlib.md5(row_dict['device_key'].encode()).hexdigest()[:6].upper()}"
        
        scan_summary = {
            "id": row_dict["id"],
            "patient_id": patient_id,
            "product_name": row_dict["product_name"],
            "brand": row_dict["brand"] or "Unknown",
            "score": score,
            "verdict": row_dict["verdict"],
            "scanned_at": row_dict["scanned_at"],
            "category": row_dict["category"] or "other",
            "tier": tier
        }
        
        recent_scans.append(scan_summary)
        
        if is_glycemic_threat:
            threat_triggers = []
            cons_list = analysis.get("cons", [])
            for c in cons_list:
                if any(kw in c.lower() for kw in ["sugar", "glycemic", "maida", "maltodextrin", "insulin", "starch", "palm oil", "sodium"]):
                    threat_triggers.append(c)
            
            glycemic_threat_feed.append({
                "scan_id": row_dict["id"],
                "patient_id": patient_id,
                "product_name": scan_summary["product_name"],
                "brand": scan_summary["brand"],
                "scanned_at": scan_summary["scanned_at"],
                "sugar_content": analysis.get("sugar", 0.0),
                "sodium_content": analysis.get("sodium_mg", analysis.get("sodium", 0.0)),
                "verdict": scan_summary["verdict"],
                "threat_triggers": threat_triggers or ["Glycemic threat trigger flagged by local rules engine."]
            })
            
    # Calculate ratios
    safety_ratios = {
        "Safe": round((safe_count / total_scans * 100), 1) if total_scans > 0 else 0.0,
        "Limit": round((limit_count / total_scans * 100), 1) if total_scans > 0 else 0.0,
        "Avoid": round((avoid_count / total_scans * 100), 1) if total_scans > 0 else 0.0
    }
    
    # Check format request: HTML vs JSON
    accept_header = request.headers.get("accept", "")
    is_html_request = "text/html" in accept_header and format != "json"
    
    if format == "html" or is_html_request:
        # Build dynamic HTML blocks
        threat_html = ""
        if not glycemic_threat_feed:
            threat_html = """
            <div class="empty-state-card card">
              <div class="empty-icon">💚</div>
              <h3>All Clear!</h3>
              <p>No glycemic or safety threats have been flagged in this cohort in the last 30 days.</p>
            </div>
            """
        else:
            for threat in glycemic_threat_feed[:20]:
                triggers_html = "".join(f"<li>{t}</li>" for t in threat["threat_triggers"])
                escaped_product = threat['product_name'].replace("'", "\\'")
                escaped_verdict = threat['verdict'].replace("'", "\\'")
                threat_html += f"""
                <div class="threat-item-card">
                  <div class="threat-header">
                    <span class="threat-patient-id"><span class="patient-icon">🩺</span> {threat['patient_id']}</span>
                    <span class="threat-date">{threat['scanned_at']}</span>
                  </div>
                  <div class="threat-body">
                    <h3 class="threat-product">{threat['product_name']} <span class="threat-brand">by {threat['brand']}</span></h3>
                    
                    <div class="threat-metrics-grid">
                      <div class="threat-metric">
                        <span class="metric-lbl">Sugar Content</span>
                        <span class="metric-val">{threat['sugar_content']}g</span>
                      </div>
                      <div class="threat-metric">
                        <span class="metric-lbl">Sodium</span>
                        <span class="metric-val">{threat['sodium_content']}mg</span>
                      </div>
                      <div class="threat-metric">
                        <span class="metric-lbl">Verdict</span>
                        <span class="metric-val alert-danger">{threat['verdict']}</span>
                      </div>
                    </div>
                    
                    <div class="threat-triggers-section">
                      <div class="trigger-label">DIAGNOSTIC TRIGGERS</div>
                      <ul class="trigger-list">
                        {triggers_html}
                      </ul>
                    </div>
                  </div>
                  <div class="threat-footer">
                    <button class="btn btn-action" onclick="showInterventionModal('{threat['patient_id']}', '{escaped_product}', '{escaped_verdict}')">Draft Intervention</button>
                  </div>
                </div>
                """

        scans_html = ""
        if not recent_scans:
            scans_html = """
            <tr>
              <td colspan="7" style="text-align: center; padding: 30px;">
                <div class="empty-icon" style="font-size: 24px;">📭</div>
                <p style="font-weight: 700; margin-top: 8px;">No recent scan activity found.</p>
              </td>
            </tr>
            """
        else:
            for scan in recent_scans[:50]:
                scans_html += f"""
                <tr>
                  <td><code>{scan['patient_id']}</code></td>
                  <td class="product-cell">{scan['product_name']}</td>
                  <td>{scan['brand']}</td>
                  <td><span class="score-badge badge-{scan['tier'].lower()}">{scan['score']}/10</span></td>
                  <td>{scan['verdict']}</td>
                  <td><span class="tier-pill tier-{scan['tier'].lower()}">{scan['tier']}</span></td>
                  <td class="time-cell">{scan['scanned_at']}</td>
                </tr>
                """

        html_content = DASHBOARD_TEMPLATE
        html_content = html_content.replace("{{dietitian_name}}", dietitian["name"])
        html_content = html_content.replace("{{cohort_code}}", code)
        html_content = html_content.replace("{{total_patients}}", str(len(device_keys)))
        html_content = html_content.replace("{{total_scans_30d}}", str(total_scans))
        html_content = html_content.replace("{{threat_feed_count}}", str(len(glycemic_threat_feed)))
        html_content = html_content.replace("{{safe_ratio}}", str(safety_ratios["Safe"]))
        html_content = html_content.replace("{{limit_ratio}}", str(safety_ratios["Limit"]))
        html_content = html_content.replace("{{avoid_ratio}}", str(safety_ratios["Avoid"]))
        html_content = html_content.replace("{{threat_html}}", threat_html)
        html_content = html_content.replace("{{scans_html}}", scans_html)
        
        return HTMLResponse(content=html_content, status_code=200)

    return {
        "dietitian_name": dietitian["name"],
        "cohort_code": code,
        "total_patients": len(device_keys),
        "total_scans_30d": total_scans,
        "safety_ratios": safety_ratios,
        "glycemic_threat_feed": glycemic_threat_feed[:20],
        "recent_scans": recent_scans[:50]
    }

@router.get("/cohort")
async def get_dietitian_cohort(
    cohort_code: str = Query(...),
    key: str = Query(...)
):
    """
    Returns high-resolution clinical insights and macro breakdowns for a specific cohort.
    Ensures that the requesting clinician is authorized using their dietitian_key.
    """
    with db_conn() as conn:
        dietitian = conn.execute("SELECT * FROM dietitians WHERE dietitian_key=? AND code=?", (key.strip(), cohort_code.strip().upper())).fetchone()
        if not dietitian:
            raise HTTPException(status_code=403, detail="Unauthorized cohort access.")
        
        # Fetch patient cohort details
        patients = conn.execute("SELECT device_key FROM patient_cohorts WHERE dietitian_code=?", (cohort_code.strip().upper(),)).fetchall()
        device_keys = [p["device_key"] for p in patients]
        
        if not device_keys:
            return {
                "cohort_code": cohort_code,
                "patients": []
            }
            
        patient_insights = []
        for dk in device_keys:
            patient_id = f"PATIENT_{hashlib.md5(dk.encode()).hexdigest()[:6].upper()}"
            # Fetch patient devices metadata (streak, persona, last scan date)
            dev_row = conn.execute("SELECT * FROM devices WHERE device_key=?", (dk,)).fetchone()
            dev_dict = dict(dev_row) if dev_row else {}
            
            # Fetch aggregate scans count
            scans_count = conn.execute("SELECT COUNT(*) as c FROM scans WHERE device_key=?", (dk,)).fetchone()["c"]
            
            # Fetch daily meal log aggregates
            from app.database.connection import get_daily_macro_totals
            macros = get_daily_macro_totals(dk)
            
            # Fetch 30-day CGM telemetry for time-in-range (TIR) and eA1c calculations
            cgm_rows = conn.execute(
                """
                SELECT glucose_mgdl 
                FROM cgm_readings 
                WHERE device_key=? AND datetime(recorded_at) >= datetime('now', '-30 days')
                """,
                (dk,)
            ).fetchall()
            
            cgm_stats = None
            if cgm_rows:
                cgm_total = len(cgm_rows)
                cgm_sum = sum(cr["glucose_mgdl"] for cr in cgm_rows)
                cgm_in_range = sum(1 for cr in cgm_rows if 70 <= cr["glucose_mgdl"] <= 140)
                cgm_avg = round(cgm_sum / cgm_total, 1)
                cgm_stats = {
                    "average_glucose_mgdl": cgm_avg,
                    "estimated_hba1c": round((cgm_avg + 46.7) / 28.7, 2),
                    "time_in_range_percent": round((cgm_in_range / cgm_total) * 100, 1)
                }

            patient_insights.append({
                "patient_id": patient_id,
                "persona": dev_dict.get("persona", "general"),
                "streak_days": dev_dict.get("streak_days", 0),
                "last_scan_date": dev_dict.get("last_scan_date", ""),
                "scans_count": scans_count,
                "today_macros": macros,
                "cgm_stats": cgm_stats
            })
            
    return {
        "cohort_code": cohort_code,
        "total_active_patients": len(device_keys),
        "patients": patient_insights
    }

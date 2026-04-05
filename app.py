"""
Cyberbullying Detection System — Premium Streamlit Application
Author: Mir Shahadut Hossain
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import pickle
import joblib
import warnings
from datetime import datetime
 
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberGuard · Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — DARK PREMIUM THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base reset ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');
 
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}
 
/* Remove Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header {
    visibility: visible;
    background: transparent !important;
}
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
}
.block-container { padding: 0 2rem 2rem 2rem; max-width: 1400px; }
 
/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 2px; }
 
/* ── App background ── */
.stApp {
    background: #070b14;
    background-image:
        radial-gradient(ellipse 80% 50% at 10% 0%, rgba(59,130,246,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(139,92,246,0.07) 0%, transparent 60%);
}
 
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid rgba(51,65,85,0.6) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.85rem;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.8rem;
    color: #64748b;
}
 
/* ── Headings ── */
.sidebar-brand-wrap {
    padding: 1.1rem 0 0.55rem;
}
.sidebar-brand-card {
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.7rem;
    text-align: center;
    padding: 1.2rem 1rem 1.05rem;
    background:
        radial-gradient(circle at top, rgba(96,165,250,0.18), transparent 55%),
        linear-gradient(145deg, rgba(15,23,42,0.98), rgba(17,24,39,0.94));
    border: 1px solid rgba(96,165,250,0.22);
    border-radius: 20px;
    box-shadow: 0 18px 40px rgba(2,6,23,0.28), inset 0 1px 0 rgba(255,255,255,0.04);
}
.sidebar-brand-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #38bdf8, #8b5cf6, #22c55e);
}
.sidebar-brand-icon {
    width: 48px !important;
    height: 48px !important;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem !important;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.18), 0 12px 24px rgba(37,99,235,0.25);
}
.sidebar-brand-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.03em;
    color: #f8fafc;
}
.sidebar-brand-subtitle {
    font-size: 0.72rem;
    color: #94a3b8;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
.sidebar-credentials-wrap {
    margin-top: 1rem;
    padding-bottom: 0.35rem;
}
.sidebar-credentials-card {
    position: relative;
    overflow: hidden;
    padding: 1rem 0.95rem 0.95rem;
    background:
        radial-gradient(circle at top right, rgba(56,189,248,0.12), transparent 45%),
        linear-gradient(160deg, rgba(15,23,42,0.98), rgba(11,18,32,0.96));
    border: 1px solid rgba(71,85,105,0.45);
    border-radius: 18px;
    box-shadow: 0 18px 36px rgba(2,6,23,0.24), inset 0 1px 0 rgba(255,255,255,0.03);
}
.sidebar-credentials-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #22c55e, #38bdf8, #8b5cf6);
}
.sidebar-credentials-label {
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 0.45rem;
}
.sidebar-credentials-name {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    line-height: 1.15;
    color: #f8fafc;
    font-weight: 700;
    margin-bottom: 0.8rem;
}
.sidebar-credentials-link {
    display: block;
    text-decoration: none;
    color: #cbd5e1 !important;
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(51,65,85,0.75);
    border-radius: 12px;
    padding: 0.7rem 0.8rem;
    margin-bottom: 0.55rem;
    transition: all 0.2s ease;
}
.sidebar-credentials-link:last-child {
    margin-bottom: 0;
}
.sidebar-credentials-link:hover {
    border-color: rgba(56,189,248,0.45);
    background: rgba(15,23,42,0.92);
    transform: translateY(-1px);
}
.sidebar-credentials-link strong {
    display: block;
    font-size: 0.66rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.2rem;
}
.sidebar-credentials-link span {
    display: block;
    font-size: 0.72rem;
    line-height: 1.35;
    color: #cbd5e1;
    word-break: break-word;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    text-shadow:
        0 0 8px rgba(255,255,255,0.18),
        0 0 18px rgba(56,189,248,0.22),
        0 0 34px rgba(59,130,246,0.14);
}
 
/* ── Metric cards ── */
.metric-card {
    background:
        radial-gradient(circle at top left, rgba(56,189,248,0.12), transparent 34%),
        linear-gradient(145deg, rgba(15,23,42,0.96), rgba(17,24,39,0.92));
    border: 1px solid rgba(71,85,105,0.58);
    border-radius: 18px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 18px 36px rgba(2,6,23,0.28), inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter: blur(14px) saturate(135%);
    -webkit-backdrop-filter: blur(14px) saturate(135%);
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, linear-gradient(90deg, #3b82f6, #8b5cf6));
}
.metric-card::after {
    content: '';
    position: absolute;
    inset: auto -20% -35% auto;
    width: 160px;
    height: 160px;
    background: radial-gradient(circle, rgba(59,130,246,0.14), transparent 70%);
    filter: blur(10px);
    pointer-events: none;
}
.metric-card:hover {
    border-color: rgba(96,165,250,0.55);
    box-shadow: 0 22px 44px rgba(2,6,23,0.34), inset 0 1px 0 rgba(255,255,255,0.06), 0 0 24px rgba(59,130,246,0.16);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.metric-sub {
    font-size: 0.75rem;
    color: #475569;
}
 
/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1.25rem;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #3b82f6;
    flex-shrink: 0;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 600;
}
 
/* ── Content panels ── */
.panel {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at top right, rgba(56,189,248,0.08), transparent 30%),
        linear-gradient(150deg, rgba(13,17,23,0.96), rgba(15,23,42,0.9));
    border: 1px solid rgba(71,85,105,0.42);
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 16px 34px rgba(2,6,23,0.24), inset 0 1px 0 rgba(255,255,255,0.03);
    backdrop-filter: blur(10px) saturate(125%);
    -webkit-backdrop-filter: blur(10px) saturate(125%);
}
.panel::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.55), rgba(139,92,246,0.4), rgba(34,197,94,0.28));
}
.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
}
 
/* ── Prediction result boxes ── */
.result-safe {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at top right, rgba(34,197,94,0.14), transparent 32%),
        linear-gradient(145deg, rgba(6,78,59,0.2), rgba(15,23,42,0.94));
    border: 1px solid rgba(52,211,153,0.34);
    border-radius: 18px;
    padding: 1.75rem 2rem;
    box-shadow: 0 18px 36px rgba(2,6,23,0.26), 0 0 24px rgba(16,185,129,0.08);
}
.result-bullying {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at top right, rgba(248,113,113,0.14), transparent 32%),
        linear-gradient(145deg, rgba(127,29,29,0.22), rgba(15,23,42,0.94));
    border: 1px solid rgba(248,113,113,0.34);
    border-radius: 18px;
    padding: 1.75rem 2rem;
    box-shadow: 0 18px 36px rgba(2,6,23,0.26), 0 0 24px rgba(239,68,68,0.08);
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.result-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 0.4rem;
}
 
/* ── Confidence bar ── */
.conf-track {
    background: rgba(30,41,59,0.8);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin-top: 0.4rem;
}
.conf-fill {
    height: 8px;
    border-radius: 99px;
    transition: width 0.5s ease;
}
 
/* ── Word chips ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.75rem; }
.chip {
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 99px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #93c5fd;
    font-family: 'DM Sans', sans-serif;
}
.chip-red {
    background: rgba(239,68,68,0.1);
    border-color: rgba(239,68,68,0.25);
    color: #fca5a5;
}
 
/* ── Tag badge ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-blue  { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.2); }
.badge-green { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.2); }
.badge-red   { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.2); }
.badge-amber { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.2); }
 
/* ── Hero section ── */
.hero {
    position: relative;
    overflow: hidden;
    padding: 2.6rem 2.4rem 2.2rem;
    margin-bottom: 1.6rem;
    border-radius: 30px;
    text-align: center;
    background:
        radial-gradient(circle at top left, rgba(56,189,248,0.16), transparent 34%),
        radial-gradient(circle at bottom right, rgba(34,197,94,0.12), transparent 30%),
        linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.92) 48%, rgba(12,18,32,0.98));
    border: 1px solid rgba(71,85,105,0.5);
    box-shadow: 0 26px 60px rgba(2,6,23,0.34), inset 0 1px 0 rgba(255,255,255,0.04);
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        linear-gradient(120deg, rgba(59,130,246,0.08), transparent 30%),
        linear-gradient(300deg, rgba(139,92,246,0.08), transparent 28%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #38bdf8, #3b82f6, #8b5cf6, #22c55e);
    opacity: 0.95;
}
.hero-eyebrow {
    position: relative;
    z-index: 1;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #7dd3fc;
    font-weight: 700;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 28px;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, #38bdf8, #8b5cf6);
}
.hero-title {
    position: relative;
    z-index: 1;
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 4.6vw, 3.7rem);
    font-weight: 800;
    line-height: 1.02;
    color: #f8fafc;
    margin-bottom: 1rem;
    letter-spacing: -0.04em;
    text-shadow:
        0 0 10px rgba(255,255,255,0.26),
        0 0 24px rgba(56,189,248,0.34),
        0 0 46px rgba(59,130,246,0.22),
        0 0 72px rgba(59,130,246,0.12);
}
.hero-title span {
    color: #7dd3fc;
    text-shadow:
        0 0 12px rgba(255,255,255,0.22),
        0 0 26px rgba(56,189,248,0.4),
        0 0 54px rgba(56,189,248,0.2);
}
.hero-sub {
    position: relative;
    z-index: 1;
    font-size: 1rem;
    color: #cbd5e1;
    max-width: 680px;
    line-height: 1.75;
    margin: 0 auto;
}
@media (max-width: 768px) {
    .hero {
        padding: 2rem 1.3rem 1.7rem;
        border-radius: 24px;
    }
    .hero-eyebrow {
        letter-spacing: 0.14em;
    }
    .hero-sub {
        font-size: 0.95rem;
    }
}
 
/* ── Feature list ── */
.feature-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.875rem 0;
    border-bottom: 1px solid rgba(30,41,59,0.6);
}
.feature-item:last-child { border-bottom: none; }
.feature-icon {
    width: 28px; height: 28px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-top: 1px;
}
.feature-text { font-size: 0.875rem; color: #94a3b8; line-height: 1.5; }
.feature-text strong { color: #e2e8f0; font-weight: 500; }
 
/* ── Data table ── */
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.table-scroll {
    max-height: 520px;
    overflow-y: auto;
    overflow-x: hidden;
    border: 1px solid rgba(71,85,105,0.4);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(15,23,42,0.5), rgba(8,15,30,0.38));
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 0 20px rgba(59,130,246,0.06);
}
.styled-table th {
    background: rgba(15,23,42,0.8);
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 10px 14px;
    border-bottom: 1px solid rgba(51,65,85,0.5);
    text-align: left;
    position: sticky;
    top: 0;
    z-index: 1;
}
.styled-table td {
    padding: 9px 14px;
    border-bottom: 1px solid rgba(30,41,59,0.4);
    color: #94a3b8;
}
.styled-table tr:last-child td { border-bottom: none; }
.styled-table tr:hover td { background: rgba(30,41,59,0.3); color: #e2e8f0; }
 
div[data-testid="stSidebar"] div[style*='display:flex;align-items:center;gap:10px;margin-bottom:1rem;'] {
    position: relative;
    overflow: hidden;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.65rem !important;
    text-align: center !important;
    padding: 1.2rem 1rem 1rem !important;
    margin-bottom: 1rem !important;
    background:
        radial-gradient(circle at top, rgba(96,165,250,0.18), transparent 55%),
        linear-gradient(145deg, rgba(15,23,42,0.98), rgba(17,24,39,0.94)) !important;
    border: 1px solid rgba(96,165,250,0.2) !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 45px rgba(2,6,23,0.28), inset 0 1px 0 rgba(255,255,255,0.03) !important;
}
div[data-testid="stSidebar"] div[style*='display:flex;align-items:center;gap:10px;margin-bottom:1rem;']::before {
    content: '';
    position: absolute;
    inset: 0 auto auto 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, #38bdf8, #8b5cf6, #22c55e);
    opacity: 0.95;
}
div[data-testid="stSidebar"] div[style*='display:flex;align-items:center;gap:10px;margin-bottom:1rem;'] > div:first-child {
    width: 48px !important;
    height: 48px !important;
    border-radius: 16px !important;
    font-size: 1.15rem !important;
}
div[data-testid="stSidebar"] div[style*='display:flex;align-items:center;gap:10px;margin-bottom:1rem;'] > div:last-child {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}
div[data-testid="stSidebar"] div[style*='font-family:Syne,sans-serif;font-size:0.9rem;font-weight:700;color:#f1f5f9;'] {
    font-size: 1.1rem !important;
    line-height: 1.05 !important;
    letter-spacing: -0.03em !important;
}
div[data-testid="stSidebar"] div[style*='font-size:0.65rem;color:#475569;letter-spacing:0.05em;'] {
    font-size: 0.72rem !important;
    color: #94a3b8 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
}

/* ── File status ── */
.file-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1rem;
    background: linear-gradient(145deg, rgba(15,23,42,0.82), rgba(17,24,39,0.72));
    border: 1px solid rgba(71,85,105,0.34);
    border-radius: 14px;
    box-shadow: 0 10px 24px rgba(2,6,23,0.18), inset 0 1px 0 rgba(255,255,255,0.03);
    margin-bottom: 6px;
    font-size: 0.8rem;
}
.file-name { color: #94a3b8; font-family: monospace; }
.file-ok   { color: #34d399; font-size: 0.7rem; }
.file-miss { color: #f87171; font-size: 0.7rem; }
 
/* ── Streamlit widget overrides ── */
@property --box-angle {
    syntax: "<angle>";
    inherits: false;
    initial-value: 0deg;
}

.metric-card,
.panel,
.result-safe,
.result-bullying,
.table-scroll,
.file-row {
    --box-angle: 0deg;
    position: relative;
    border: 1px solid transparent !important;
    animation: neonBorderSpin 8s linear infinite, neonBoxPulse 3.6s ease-in-out infinite;
}

.metric-card {
    background:
        radial-gradient(circle at top left, rgba(56,189,248,0.14), transparent 34%),
        linear-gradient(145deg, rgba(15,23,42,0.96), rgba(17,24,39,0.92)) padding-box,
        conic-gradient(from var(--box-angle), rgba(56,189,248,0.88), rgba(124,58,237,0.9), rgba(34,211,238,0.86), rgba(96,165,250,0.82), rgba(56,189,248,0.88)) border-box;
}

.panel {
    background:
        radial-gradient(circle at top right, rgba(56,189,248,0.1), transparent 30%),
        linear-gradient(150deg, rgba(13,17,23,0.96), rgba(15,23,42,0.9)) padding-box,
        conic-gradient(from var(--box-angle), rgba(56,189,248,0.62), rgba(139,92,246,0.56), rgba(34,197,94,0.42), rgba(56,189,248,0.62)) border-box;
}

.result-safe {
    background:
        radial-gradient(circle at top right, rgba(34,197,94,0.16), transparent 32%),
        linear-gradient(145deg, rgba(6,78,59,0.24), rgba(15,23,42,0.94)) padding-box,
        conic-gradient(from var(--box-angle), rgba(34,197,94,0.88), rgba(45,212,191,0.84), rgba(56,189,248,0.78), rgba(34,197,94,0.88)) border-box;
}

.result-bullying {
    background:
        radial-gradient(circle at top right, rgba(248,113,113,0.16), transparent 32%),
        linear-gradient(145deg, rgba(127,29,29,0.24), rgba(15,23,42,0.94)) padding-box,
        conic-gradient(from var(--box-angle), rgba(248,113,113,0.9), rgba(236,72,153,0.84), rgba(139,92,246,0.8), rgba(248,113,113,0.9)) border-box;
}

.table-scroll {
    background:
        linear-gradient(180deg, rgba(15,23,42,0.54), rgba(8,15,30,0.42)) padding-box,
        conic-gradient(from var(--box-angle), rgba(56,189,248,0.62), rgba(99,102,241,0.54), rgba(34,211,238,0.46), rgba(56,189,248,0.62)) border-box;
}

.file-row {
    background:
        linear-gradient(145deg, rgba(15,23,42,0.82), rgba(17,24,39,0.72)) padding-box,
        conic-gradient(from var(--box-angle), rgba(56,189,248,0.52), rgba(139,92,246,0.42), rgba(34,197,94,0.34), rgba(56,189,248,0.52)) border-box;
}

.metric-card:hover,
.panel:hover,
.result-safe:hover,
.result-bullying:hover,
.table-scroll:hover,
.file-row:hover {
    box-shadow:
        0 22px 44px rgba(2,6,23,0.38),
        inset 0 1px 0 rgba(255,255,255,0.07),
        0 0 28px rgba(56,189,248,0.22),
        0 0 48px rgba(139,92,246,0.12) !important;
    transform: translateY(-2px);
}

@keyframes neonBorderSpin {
    from { --box-angle: 0deg; }
    to { --box-angle: 360deg; }
}

@keyframes neonBoxPulse {
    0%, 100% {
        box-shadow:
            0 16px 34px rgba(2,6,23,0.24),
            inset 0 1px 0 rgba(255,255,255,0.04),
            0 0 16px rgba(56,189,248,0.08);
    }
    50% {
        box-shadow:
            0 20px 40px rgba(2,6,23,0.3),
            inset 0 1px 0 rgba(255,255,255,0.06),
            0 0 24px rgba(56,189,248,0.16),
            0 0 42px rgba(139,92,246,0.1);
    }
}

.stTextArea textarea {
    background: #0d1117 !important;
    border: 1px solid rgba(51,65,85,0.6) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: rgba(59,130,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}
.stButton > button,
div[data-testid="stButton"] > button,
div[data-testid="stButton"] button {
    position: relative !important;
    isolation: isolate !important;
    overflow: hidden !important;
    background: linear-gradient(135deg, rgba(8,15,30,0.72), rgba(18,26,46,0.58)) !important;
    color: #eff6ff !important;
    border: 1px solid rgba(125,211,252,0.18) !important;
    border-radius: 18px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 0.78rem 1.5rem !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
    letter-spacing: 0.03em !important;
    text-shadow:
        0 0 10px rgba(255,255,255,0.2),
        0 0 22px rgba(96,165,250,0.28),
        0 0 42px rgba(139,92,246,0.16) !important;
    box-shadow:
        0 14px 32px rgba(2,6,23,0.32),
        inset 0 1px 0 rgba(255,255,255,0.12),
        0 0 18px rgba(59,130,246,0.18) !important;
    backdrop-filter: blur(14px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(14px) saturate(140%) !important;
    animation: ctaPulse 2.8s ease-in-out infinite !important;
}
.stButton > button::before,
div[data-testid="stButton"] > button::before,
div[data-testid="stButton"] button::before {
    content: "" !important;
    position: absolute !important;
    inset: -1px !important;
    border-radius: inherit !important;
    padding: 2px !important;
    background: conic-gradient(from 0deg, #38bdf8, #7c3aed, #22d3ee, #60a5fa, #38bdf8) !important;
    -webkit-mask:
        linear-gradient(#000 0 0) content-box,
        linear-gradient(#000 0 0) !important;
    -webkit-mask-composite: xor !important;
    mask:
        linear-gradient(#000 0 0) content-box,
        linear-gradient(#000 0 0) !important;
    mask-composite: exclude !important;
    animation: ctaSpin 4s linear infinite !important;
    z-index: -1 !important;
}
.stButton > button::after,
div[data-testid="stButton"] > button::after,
div[data-testid="stButton"] button::after {
    content: "" !important;
    position: absolute !important;
    inset: 14% !important;
    border-radius: inherit !important;
    background: radial-gradient(circle, rgba(56,189,248,0.3), rgba(139,92,246,0.12) 45%, transparent 72%) !important;
    filter: blur(18px) !important;
    opacity: 0.9 !important;
    z-index: -1 !important;
}
.stButton > button:hover,
div[data-testid="stButton"] > button:hover,
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    border-color: rgba(125,211,252,0.35) !important;
    box-shadow:
        0 18px 38px rgba(2,6,23,0.42),
        inset 0 1px 0 rgba(255,255,255,0.16),
        0 0 26px rgba(56,189,248,0.32),
        0 0 50px rgba(139,92,246,0.18) !important;
}
.stButton > button:hover::after,
div[data-testid="stButton"] > button:hover::after,
div[data-testid="stButton"] button:hover::after {
    opacity: 1 !important;
    filter: blur(22px) !important;
}

[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] div[data-testid="stButton"] > button,
[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background: rgba(15,23,42,0.82) !important;
    border: 1px solid rgba(51,65,85,0.55) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 0.85rem !important;
    letter-spacing: 0.02em !important;
    text-shadow: none !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    animation: none !important;
}
[data-testid="stSidebar"] .stButton > button::before,
[data-testid="stSidebar"] .stButton > button::after,
[data-testid="stSidebar"] div[data-testid="stButton"] > button::before,
[data-testid="stSidebar"] div[data-testid="stButton"] > button::after,
[data-testid="stSidebar"] div[data-testid="stButton"] button::before,
[data-testid="stSidebar"] div[data-testid="stButton"] button::after {
    content: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover,
[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    transform: none !important;
    box-shadow: none !important;
}

@keyframes ctaSpin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
@keyframes ctaPulse {
    0%, 100% {
        box-shadow:
            0 14px 32px rgba(2,6,23,0.32),
            inset 0 1px 0 rgba(255,255,255,0.12),
            0 0 18px rgba(59,130,246,0.18);
    }
    50% {
        box-shadow:
            0 18px 38px rgba(2,6,23,0.38),
            inset 0 1px 0 rgba(255,255,255,0.14),
            0 0 28px rgba(56,189,248,0.26),
            0 0 46px rgba(139,92,246,0.14);
    }
}
.stSelectbox > div > div {
    background: #0d1117 !important;
    border: 1px solid rgba(51,65,85,0.6) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
[data-testid="stMetric"] {
    background: transparent !important;
}
div[data-testid="stSidebarNav"] { display: none; }
 
/* ── Plotly charts dark bg ── */
.js-plotly-plot { background: transparent !important; }
.plot-container { background: transparent !important; }
 
/* ── Divider ── */
.fancy-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(51,65,85,0.6), transparent);
    margin: 1.5rem 0;
}
 
/* ── Sidebar nav ── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.65rem 0.875rem;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s;
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 2px;
    text-decoration: none;
}
.nav-item.active {
    background: rgba(59,130,246,0.12);
    color: #60a5fa;
    border: 1px solid rgba(59,130,246,0.2);
}
.nav-item:hover:not(.active) { background: rgba(30,41,59,0.5); color: #94a3b8; }
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "input_text": "",
        "prediction_result": None,
        "history": [],
        "page": "Overview",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
 
init_session_state()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ASSET LOADING — safe, cached
# ─────────────────────────────────────────────────────────────────────────────
FILE_MAP = {
    "model":      "cyberbullying_model_lr.pkl",
    "vectorizer": "tfidf_vectorizer.pkl",
    "dataset":    "aggression_parsed_dataset.csv",
    "X_test":     "X_test_sparse.npz",
    "y_test":     "y_test.npy",
}
 
def safe_load_pkl(path: str):
    """Try joblib first, fall back to pickle."""
    try:
        return joblib.load(path)
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
 
@st.cache_resource(show_spinner=False)
def load_model():
    return safe_load_pkl(FILE_MAP["model"])
 
@st.cache_resource(show_spinner=False)
def load_vectorizer():
    return safe_load_pkl(FILE_MAP["vectorizer"])
 
@st.cache_data(show_spinner=False)
def load_dataset():
    path = FILE_MAP["dataset"]
    if not os.path.exists(path):
        return None, None, None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, None, None
    # Infer text / label columns
    text_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["text","comment","tweet","content","message"])),
        df.select_dtypes(include="object").columns[0] if len(df.select_dtypes(include="object").columns) else None
    )
    label_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["label","class","target","category","oh_"])),
        None
    )
    return df, text_col, label_col
 
@st.cache_data(show_spinner=False)
def load_test_split():
    import scipy.sparse as sp
    X, y = None, None
    try:
        if os.path.exists(FILE_MAP["X_test"]):
            X = sp.load_npz(FILE_MAP["X_test"])
    except Exception:
        pass
    try:
        if os.path.exists(FILE_MAP["y_test"]):
            y = np.load(FILE_MAP["y_test"])
    except Exception:
        pass
    return X, y
 
model      = load_model()
vectorizer = load_vectorizer()
df, text_col, label_col = load_dataset()
X_test, y_test = load_test_split()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# NLP UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("omw-1.4",   quiet=True)
    _STOP = set(stopwords.words("english"))
    _LEM  = WordNetLemmatizer()
    NLTK_OK = True
except Exception:
    NLTK_OK = False
 
def clean_text(text: str) -> str:
    """Reproduce the same cleaning used during training."""
    t = str(text).lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#", "", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if NLTK_OK:
        t = " ".join(_LEM.lemmatize(w) for w in t.split() if w not in _STOP)
    return t
 
def get_top_features(text: str, n: int = 10):
    """Return (bullying_words, safe_words) from TF-IDF × LR coefficients."""
    if model is None or vectorizer is None:
        return [], []
    if not (hasattr(model, "coef_") and hasattr(vectorizer, "transform")):
        return [], []
    try:
        vec     = vectorizer.transform([clean_text(text)])
        coefs   = model.coef_[0]
        feat    = vectorizer.get_feature_names_out()
        _, cols = vec.nonzero()
        scores  = [(feat[c], coefs[c] * vec[0, c]) for c in cols]
        scores.sort(key=lambda x: x[1], reverse=True)
        bully = [w for w, s in scores if s > 0][:n]
        safe  = [w for w, s in scores if s < 0][:n]
        return bully, safe
    except Exception:
        return [], []
 
def run_prediction(text: str):
    """Return dict with prediction, confidence, proba_cb, proba_ok."""
    if not text.strip():
        return None
    cleaned = clean_text(text)
 
    if model is not None and vectorizer is not None:
        try:
            X   = vectorizer.transform([cleaned])
            raw = int(model.predict(X)[0])
            if hasattr(model, "predict_proba"):
                proba    = model.predict_proba(X)[0]
                proba_cb = float(proba[1]) if len(proba) > 1 else float(proba[0])
                proba_ok = 1.0 - proba_cb
            else:
                proba_cb = 0.9 if raw == 1 else 0.1
                proba_ok = 1.0 - proba_cb
            return {
                "label":    raw,
                "conf":     proba_cb if raw == 1 else proba_ok,
                "proba_cb": proba_cb,
                "proba_ok": proba_ok,
                "source":   "model",
            }
        except Exception:
            pass
 
    # Fallback keyword heuristic
    KB = ["stupid","idiot","dumb","hate","kill","die","loser","worthless","ugly","fat","nobody","moron","freak"]
    hits = sum(1 for w in KB if w in text.lower())
    if hits >= 2:
        c = min(0.90, 0.60 + hits * 0.08)
        raw = 1
    else:
        c   = 0.15
        raw = 0
    return {
        "label":    raw,
        "conf":     c,
        "proba_cb": c if raw == 1 else 1 - c,
        "proba_ok": 1 - c if raw == 1 else c,
        "source":   "heuristic",
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_metrics():
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    if model is None or X_test is None or y_test is None:
        return None
    try:
        y_pred = model.predict(X_test)
        return {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "cm":        confusion_matrix(y_test, y_pred),
            "report":    classification_report(y_test, y_pred, zero_division=0),
            "n_test":    len(y_test),
        }
    except Exception:
        return None
 
FALLBACK_METRICS = {
    "accuracy": 0.810, "precision": 0.964,
    "recall": 0.802,   "f1": 0.876,
    "cm": np.array([[1300,231],[1548,6287]]),
    "n_test": 9366,    "source": "fallback",
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-brand-wrap'>
        <div class='sidebar-brand-card'>
            <div class='sidebar-brand-icon' style='width:32px;height:32px;background:linear-gradient(135deg,#1d4ed8,#7c3aed);
                        border-radius:8px;display:flex;align-items:center;justify-content:center;
                        font-size:1rem;'>🛡️</div>
            <div>
                <div class='sidebar-brand-title'>CyberGuard</div>
                <div class='sidebar-brand-subtitle'>Detection System v2.0</div>
            </div>
        </div>
    </div>
    <hr style='border:none;border-top:1px solid rgba(51,65,85,0.4);margin-bottom:1rem;'>
    <div style='font-size:0.6rem;letter-spacing:0.12em;text-transform:uppercase;color:#334155;
                font-weight:600;padding:0 0.25rem;margin-bottom:0.5rem;'>Navigation</div>
    """, unsafe_allow_html=True)
 
    pages = {
        "Overview":      "◈",
        "Live Detection":"◉",
        "Analytics":     "◐",
        "System Info":   "◎",
        "About":         "○",
    }
    for name, icon in pages.items():
        active = st.session_state.page == name
        style = ("background:rgba(59,130,246,0.12);color:#60a5fa;"
                 "border:1px solid rgba(59,130,246,0.2);") if active else ""
        if st.sidebar.button(f"{icon}  {name}", key=f"nav_{name}",
                              use_container_width=True):
            st.session_state.page = name
            st.rerun()
 
    st.markdown("<hr style='border:none;border-top:1px solid rgba(51,65,85,0.4);margin:1rem 0;'>",
                unsafe_allow_html=True)
 
    # Status indicators
    m_ok  = model      is not None
    v_ok  = vectorizer is not None
    d_ok  = df         is not None
    ts_ok = X_test     is not None
 
    st.markdown(f"""
    <div style='font-size:0.6rem;letter-spacing:0.12em;text-transform:uppercase;color:#334155;
                font-weight:600;padding:0 0.25rem;margin-bottom:0.5rem;'>System Status</div>
    <div style='background:#0a0f1a;border-radius:8px;padding:0.75rem;font-size:0.75rem;'>
        <div style='display:flex;justify-content:space-between;padding:4px 0;'>
            <span style='color:#475569;'>Model</span>
            <span style='color:{"#34d399" if m_ok else "#f87171"};'>
                {"● Loaded" if m_ok else "● Missing"}
            </span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:4px 0;'>
            <span style='color:#475569;'>Vectorizer</span>
            <span style='color:{"#34d399" if v_ok else "#f87171"};'>
                {"● Loaded" if v_ok else "● Missing"}
            </span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:4px 0;'>
            <span style='color:#475569;'>Dataset</span>
            <span style='color:{"#34d399" if d_ok else "#f87171"};'>
                {"● Loaded" if d_ok else "● Missing"}
            </span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:4px 0;'>
            <span style='color:#475569;'>Test Split</span>
            <span style='color:{"#34d399" if ts_ok else "#f87171"};'>
                {"● Loaded" if ts_ok else "● Missing"}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
 
    n_hist = len(st.session_state.history)
    st.markdown(f"""
    <div style='background:#0a0f1a;border-radius:8px;padding:0.75rem;
                font-size:0.75rem;text-align:center;'>
        <div style='color:#3b82f6;font-family:Syne,sans-serif;font-size:1.25rem;
                    font-weight:700;'>{n_hist}</div>
        <div style='color:#475569;font-size:0.7rem;margin-top:2px;'>analyses this session</div>
    </div>
    """, unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='sidebar-credentials-wrap'>
        <div class='sidebar-credentials-card'>
            <div class='sidebar-credentials-label'>Created By</div>
            <div class='sidebar-credentials-name'>Mir Shahadut Hossain</div>
            <a class='sidebar-credentials-link' href='https://github.com/doyancha' target='_blank'>
                <strong>GitHub</strong>
                <span>github.com/doyancha</span>
            </a>
            <a class='sidebar-credentials-link' href='https://www.linkedin.com/in/mir-shahadut-hossain/' target='_blank'>
                <strong>LinkedIn</strong>
                <span>mir-shahadut-hossain</span>
            </a>
            <a class='sidebar-credentials-link' href='mailto:sujon6901@gmail.com'>
                <strong>Email</strong>
                <span>sujon6901@gmail.com</span>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == "Overview":
 
    # Hero
    st.markdown("""
    <div class='hero'>
        <div class='hero-eyebrow'>NLP · Machine Learning · Safety AI</div>
        <div class='hero-title'>Detect Harmful Content<br><span>Before It Causes Harm</span></div>
        <div class='hero-sub'>
            A production-grade cyberbullying detection system powered by Logistic Regression
            and TF-IDF — trained on 46,828 social media samples across 6 harm categories.
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # Metrics row
    metrics = compute_metrics() or FALLBACK_METRICS
    c1, c2, c3, c4, c5 = st.columns(5)
    cols_data = [
        (c1, f"{metrics['accuracy']*100:.1f}%",  "Accuracy",   "#3b82f6"),
        (c2, f"{metrics['precision']*100:.1f}%", "Precision",  "#8b5cf6"),
        (c3, f"{metrics['recall']*100:.1f}%",    "Recall",     "#10b981"),
        (c4, f"{metrics['f1']*100:.1f}%",        "F1-Score",   "#f59e0b"),
        (c5, "46,828",                            "Train Samples","#6366f1"),
    ]
    for col, val, lbl, color in cols_data:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='--accent:linear-gradient(90deg,{color},{color}44);'>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-value' style='color:{color};'>{val}</div>
            </div>
            """, unsafe_allow_html=True)
 
    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
 
    col_l, col_r = st.columns([1.15, 1])
 
    with col_l:
        st.markdown("""
        <div class='section-header'>
            <div class='section-dot'></div>
            <div class='section-title'>What this system does</div>
        </div>
        <div class='panel'>
        """, unsafe_allow_html=True)
 
        features = [
            ("🔍", "Real-time Detection",
             "Classify any text as <strong>cyberbullying or safe</strong> with probability scores."),
            ("📊", "6-Class Coverage",
             "Covers <strong>Religion, Age, Gender, Ethnicity, Other CB</strong> and Not-CB."),
            ("🧠", "Feature Explainability",
             "Surfaces <strong>top TF-IDF terms</strong> driving each classification decision."),
            ("⚖️", "SMOTE Balanced",
             "Class imbalance corrected via <strong>SMOTE oversampling</strong> — no recall bias."),
            ("🎛️", "GridSearchCV Tuned",
             "Hyperparameters optimised with cross-validation for <strong>best generalisation</strong>."),
        ]
        for icon, title, desc in features:
            st.markdown(f"""
            <div class='feature-item'>
                <div class='feature-icon' style='background:rgba(59,130,246,0.1);'>{icon}</div>
                <div class='feature-text'><strong>{title}</strong><br>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
 
        st.markdown("</div>", unsafe_allow_html=True)
 
    with col_r:
        st.markdown("""
        <div class='section-header'>
            <div class='section-dot' style='background:#8b5cf6;'></div>
            <div class='section-title'>Dataset snapshot</div>
        </div>
        """, unsafe_allow_html=True)
 
        if df is not None and label_col:
            try:
                dist = df[label_col].value_counts().reset_index()
                dist.columns = ["Category", "Count"]
                dist["Share"] = (dist["Count"] / dist["Count"].sum() * 100).round(1)
                colors = ["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444","#6366f1"]
                rows_html = ""
                for i, row in dist.iterrows():
                    c = colors[i % len(colors)]
                    rows_html += f"""
                    <tr>
                        <td><span style='display:inline-block;width:8px;height:8px;
                            border-radius:50%;background:{c};margin-right:8px;'></span>
                            {row['Category']}</td>
                        <td style='text-align:right;'>{row['Count']:,}</td>
                        <td style='text-align:right;'><span class='badge badge-blue'>{row['Share']}%</span></td>
                    </tr>"""
                st.markdown(f"""
                <div class='panel' style='padding:1rem;'>
                    <div class='panel-title'>Label distribution</div>
                    <div class='table-scroll'>
                        <table class='styled-table'>
                            <thead><tr><th>Category</th><th style='text-align:right;'>Samples</th>
                                <th style='text-align:right;'>Share</th></tr></thead>
                            <tbody>{rows_html}</tbody>
                        </table>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                st.info("Dataset loaded — label distribution unavailable.")
        else:
            # Fallback static table
            st.markdown("""
            <div class='panel' style='padding:1rem;'>
                <div class='panel-title'>Label distribution (reference)</div>
                <div class='table-scroll'>
                    <table class='styled-table'>
                        <thead><tr><th>Category</th><th style='text-align:right;'>Samples</th>
                            <th style='text-align:right;'>Share</th></tr></thead>
                        <tbody>
                            <tr><td>Religion</td><td style='text-align:right;'>7,995</td><td style='text-align:right;'><span class='badge badge-blue'>17.1%</span></td></tr>
                            <tr><td>Age</td><td style='text-align:right;'>7,988</td><td style='text-align:right;'><span class='badge badge-blue'>17.1%</span></td></tr>
                            <tr><td>Ethnicity</td><td style='text-align:right;'>7,955</td><td style='text-align:right;'><span class='badge badge-blue'>17.0%</span></td></tr>
                            <tr><td>Gender</td><td style='text-align:right;'>7,875</td><td style='text-align:right;'><span class='badge badge-blue'>16.8%</span></td></tr>
                            <tr><td>Not CB</td><td style='text-align:right;'>7,657</td><td style='text-align:right;'><span class='badge badge-green'>16.4%</span></td></tr>
                            <tr><td>Other CB</td><td style='text-align:right;'>7,358</td><td style='text-align:right;'><span class='badge badge-amber'>15.7%</span></td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            """, unsafe_allow_html=True)
 
        # Pipeline summary
        st.markdown("""
        <div class='panel' style='padding:1rem;margin-top:0;'>
            <div class='panel-title'>ML pipeline</div>
            <div style='display:flex;flex-wrap:wrap;gap:6px;'>
                <span class='badge badge-blue'>Raw Text</span>
                <span style='color:#334155;'>→</span>
                <span class='badge badge-blue'>Cleaning + Lemmatize</span>
                <span style='color:#334155;'>→</span>
                <span class='badge badge-blue'>TF-IDF (5k, bigrams)</span>
                <span style='color:#334155;'>→</span>
                <span class='badge badge-blue'>SMOTE</span>
                <span style='color:#334155;'>→</span>
                <span class='badge badge-blue'>GridSearchCV</span>
                <span style='color:#334155;'>→</span>
                <span class='badge badge-green'>Logistic Regression</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LIVE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "Live Detection":
 
    st.markdown("""
    <div style='padding: 2rem 0 0.5rem;'>
        <div class='hero-eyebrow'>Real-time Analysis</div>
        <div style='font-family:Syne,sans-serif;font-size:1.75rem;font-weight:700;color:#f1f5f9;margin-bottom:0.5rem;'>
            Live Detection
        </div>
        <div style='font-size:0.875rem;color:#475569;'>
            Enter any text to analyse it for cyberbullying content instantly.
        </div>
    </div>
    <hr class='fancy-divider'>
    """, unsafe_allow_html=True)
 
    EXAMPLES = {
        "— Select an example —": "",
        "Severe harassment": "You're so stupid nobody likes you, you should just disappear you ugly worthless loser",
        "Mild hostility":    "lol you're such an idiot, everyone thinks you're dumb",
        "Safe — positive":   "Great job on the presentation today! Really impressive work, well done!",
        "Safe — neutral":    "I'll be at the library around 3pm if you want to study together for the exam.",
        "Ambiguous":         "You won't last long in this field, trust me.",
    }
 
    col_main, col_side = st.columns([1.4, 1])
 
    with col_main:
        example = st.selectbox("Load example text", options=list(EXAMPLES.keys()), key="example_sel")
        if example != "— Select an example —" and EXAMPLES[example]:
            st.session_state.input_text = EXAMPLES[example]
 
        user_text = st.text_area(
            "Text to analyse",
            value=st.session_state.input_text,
            height=150,
            placeholder="Type or paste text here…",
            key="text_area_input",
            label_visibility="collapsed",
        )
        st.session_state.input_text = user_text
 
        word_count = len(user_text.split()) if user_text.strip() else 0
        char_count = len(user_text)
        st.markdown(f"""
        <div style='font-size:0.7rem;color:#334155;text-align:right;margin-top:-0.5rem;margin-bottom:0.75rem;'>
            {word_count} words · {char_count} chars
        </div>
        """, unsafe_allow_html=True)
 
        btn_col1, btn_col2, _ = st.columns([1, 1, 3])
        with btn_col1:
            analyse_clicked = st.button("🔍  Analyse", use_container_width=True)
        with btn_col2:
            if st.button("✕  Clear", use_container_width=True):
                st.session_state.input_text     = ""
                st.session_state.prediction_result = None
                st.rerun()
 
        if analyse_clicked:
            if not user_text.strip():
                st.warning("Please enter some text to analyse.")
            else:
                with st.spinner("Analysing…"):
                    result = run_prediction(user_text)
                    if result:
                        result["text"]      = user_text
                        result["timestamp"] = datetime.now().strftime("%H:%M:%S")
                        st.session_state.prediction_result = result
                        st.session_state.history.append(result)
 
        # ── Result display ──
        result = st.session_state.prediction_result
        if result:
            is_cb  = result["label"] == 1
            conf   = result["conf"]
            p_cb   = result["proba_cb"]
            p_ok   = result["proba_ok"]
 
            if is_cb:
                severity = "HIGH" if p_cb >= 0.80 else "MODERATE" if p_cb >= 0.60 else "LOW"
                sev_color= "#ef4444" if severity=="HIGH" else "#f59e0b" if severity=="MODERATE" else "#fbbf24"
                st.markdown(f"""
                <div class='result-bullying'>
                    <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;'>
                        <div class='result-title' style='color:#f87171;'>⚠ Cyberbullying Detected</div>
                        <span class='badge' style='background:rgba(239,68,68,0.15);color:{sev_color};
                            border:1px solid rgba(239,68,68,0.3);font-size:0.65rem;'>
                            {severity} RISK
                        </span>
                    </div>
                    <div class='result-label'>Confidence Score</div>
                    <div style='font-family:Syne,sans-serif;font-size:1.25rem;font-weight:700;color:#f87171;'>
                        {conf*100:.1f}%
                    </div>
                    <div class='conf-track'>
                        <div class='conf-fill' style='width:{conf*100:.1f}%;
                            background:linear-gradient(90deg,#dc2626,#f87171);'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-safe'>
                    <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;'>
                        <div class='result-title' style='color:#34d399;'>✓ Safe Content</div>
                        <span class='badge badge-green'>CLEAR</span>
                    </div>
                    <div class='result-label'>Confidence Score</div>
                    <div style='font-family:Syne,sans-serif;font-size:1.25rem;font-weight:700;color:#34d399;'>
                        {conf*100:.1f}%
                    </div>
                    <div class='conf-track'>
                        <div class='conf-fill' style='width:{conf*100:.1f}%;
                            background:linear-gradient(90deg,#059669,#34d399);'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
            # Probability breakdown
            st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='panel'>
                <div class='panel-title'>Probability Breakdown</div>
            """, unsafe_allow_html=True)
 
            for label, prob, color in [
                ("Cyberbullying", p_cb, "#ef4444"),
                ("Safe Content",  p_ok, "#10b981"),
            ]:
                st.markdown(f"""
                <div style='margin-bottom:0.75rem;'>
                    <div style='display:flex;justify-content:space-between;
                                font-size:0.8rem;color:#64748b;margin-bottom:4px;'>
                        <span>{label}</span>
                        <span style='color:{color};font-weight:500;'>{prob*100:.1f}%</span>
                    </div>
                    <div class='conf-track'>
                        <div class='conf-fill' style='width:{prob*100:.1f}%;background:{color};'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
 
            # Top features
            bully_words, safe_words = get_top_features(result["text"])
            if bully_words or safe_words:
                st.markdown("<div class='panel'><div class='panel-title'>Contributing Terms</div>",
                            unsafe_allow_html=True)
                if bully_words:
                    chips = "".join(f"<span class='chip chip-red'>{w}</span>" for w in bully_words[:8])
                    st.markdown(f"""
                    <div style='font-size:0.7rem;color:#475569;margin-bottom:4px;
                                text-transform:uppercase;letter-spacing:0.05em;'>
                        Bullying signals
                    </div>
                    <div class='chip-row'>{chips}</div>
                    """, unsafe_allow_html=True)
                if safe_words:
                    chips = "".join(f"<span class='chip'>{w}</span>" for w in safe_words[:8])
                    st.markdown(f"""
                    <div style='font-size:0.7rem;color:#475569;margin:0.75rem 0 4px;
                                text-transform:uppercase;letter-spacing:0.05em;'>
                        Neutral signals
                    </div>
                    <div class='chip-row'>{chips}</div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
 
            # Source note
            if result.get("source") == "heuristic":
                st.markdown("""
                <div style='font-size:0.7rem;color:#334155;margin-top:0.5rem;'>
                    ℹ Model files not found — using keyword heuristic fallback.
                </div>
                """, unsafe_allow_html=True)
 
    with col_side:
        # Session history
        st.markdown("""
        <div class='section-header'>
            <div class='section-dot' style='background:#8b5cf6;'></div>
            <div class='section-title'>Session History</div>
        </div>
        """, unsafe_allow_html=True)
 
        history = st.session_state.history
        if not history:
            st.markdown("""
            <div class='panel' style='text-align:center;padding:2rem 1rem;'>
                <div style='font-size:1.5rem;margin-bottom:0.5rem;'>◌</div>
                <div style='font-size:0.8rem;color:#334155;'>No analyses yet this session.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for item in reversed(history[-8:]):
                color  = "#ef4444" if item["label"] == 1 else "#10b981"
                icon   = "⚠" if item["label"] == 1 else "✓"
                status = "Cyberbullying" if item["label"] == 1 else "Safe"
                snippet = item["text"][:70] + "…" if len(item["text"]) > 70 else item["text"]
                st.markdown(f"""
                <div style='background:#0a0f1a;border:1px solid rgba(30,41,59,0.8);
                            border-radius:8px;padding:0.75rem;margin-bottom:6px;'>
                    <div style='display:flex;align-items:center;
                                justify-content:space-between;margin-bottom:4px;'>
                        <span style='font-size:0.7rem;color:{color};font-weight:500;'>
                            {icon} {status}
                        </span>
                        <span style='font-size:0.65rem;color:#334155;'>
                            {item.get("timestamp","")}
                        </span>
                    </div>
                    <div style='font-size:0.75rem;color:#475569;line-height:1.4;'>{snippet}</div>
                    <div style='font-size:0.7rem;color:#334155;margin-top:4px;'>
                        Conf: {item["conf"]*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
        # Guidance panel
        st.markdown("""
        <div class='section-header' style='margin-top:1.5rem;'>
            <div class='section-dot' style='background:#10b981;'></div>
            <div class='section-title'>Guidance</div>
        </div>
        <div class='panel' style='padding:1rem;'>
            <div class='feature-item' style='padding:0.5rem 0;'>
                <div class='feature-icon' style='background:rgba(239,68,68,0.1);font-size:0.7rem;'>80%+</div>
                <div class='feature-text' style='font-size:0.75rem;'><strong>High Risk</strong> — Immediate human review recommended</div>
            </div>
            <div class='feature-item' style='padding:0.5rem 0;'>
                <div class='feature-icon' style='background:rgba(245,158,11,0.1);font-size:0.7rem;'>60%+</div>
                <div class='feature-text' style='font-size:0.75rem;'><strong>Moderate Risk</strong> — Manual review suggested</div>
            </div>
            <div class='feature-item' style='padding:0.5rem 0;border-bottom:none;'>
                <div class='feature-icon' style='background:rgba(16,185,129,0.1);font-size:0.7rem;'>&lt;60%</div>
                <div class='feature-text' style='font-size:0.75rem;'><strong>Low / Safe</strong> — Monitor if needed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "Analytics":
    import plotly.graph_objects as go
    import plotly.express as px
 
    PLOTLY_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#94a3b8", size=11),
        margin=dict(l=10, r=10, t=30, b=10),
    )
 
    st.markdown("""
    <div style='padding: 2rem 0 0.5rem;'>
        <div class='hero-eyebrow'>Model Evaluation</div>
        <div style='font-family:Syne,sans-serif;font-size:1.75rem;font-weight:700;color:#f1f5f9;margin-bottom:0.5rem;'>
            Analytics Dashboard
        </div>
        <div style='font-size:0.875rem;color:#475569;'>
            Performance metrics, confusion analysis, and dataset insights.
        </div>
    </div>
    <hr class='fancy-divider'>
    """, unsafe_allow_html=True)
 
    metrics = compute_metrics() or FALLBACK_METRICS
    is_live = "source" not in metrics
 
    if not is_live:
        st.markdown("""
        <div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
                    border-radius:8px;padding:0.75rem 1rem;font-size:0.8rem;color:#fbbf24;
                    margin-bottom:1rem;'>
            ⚠ Model or test-split files not found — displaying reference values from training notebook.
        </div>
        """, unsafe_allow_html=True)
 
    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    m_data = [
        (c1, f"{metrics['accuracy']*100:.1f}%",  "Accuracy",  "#3b82f6"),
        (c2, f"{metrics['precision']*100:.1f}%", "Precision", "#8b5cf6"),
        (c3, f"{metrics['recall']*100:.1f}%",    "Recall",    "#10b981"),
        (c4, f"{metrics['f1']*100:.1f}%",        "F1-Score",  "#f59e0b"),
    ]
    for col, val, lbl, color in m_data:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='--accent:linear-gradient(90deg,{color},{color}44);'>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-value' style='color:{color};'>{val}</div>
            </div>
            """, unsafe_allow_html=True)
 
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
 
    # Row 1: Confusion matrix + Metric bars
    col_a, col_b = st.columns(2)
 
    with col_a:
        st.markdown("<div class='section-header'><div class='section-dot'></div><div class='section-title'>Confusion Matrix</div></div>", unsafe_allow_html=True)
        cm = metrics["cm"]
        z  = cm.tolist()
        fig_cm = go.Figure(go.Heatmap(
            z=z,
            x=["Predicted: Safe", "Predicted: CB"],
            y=["Actual: Safe", "Actual: CB"],
            text=[[str(v) for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white", "family": "Syne"},
            colorscale=[[0,"#0d1117"],[0.5,"#1d4ed8"],[1,"#3b82f6"]],
            showscale=False,
        ))
        fig_cm.update_layout(
            **PLOTLY_LAYOUT,
            height=280,
            xaxis=dict(side="bottom", tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
 
        # Derived stats
        tn, fp = int(cm[0][0]), int(cm[0][1])
        fn, tp = int(cm[1][0]), int(cm[1][1])
        st.markdown(f"""
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:-0.5rem;'>
            <div style='background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.15);
                        border-radius:8px;padding:0.6rem;text-align:center;'>
                <div style='font-size:0.65rem;color:#475569;text-transform:uppercase;'>True Positive</div>
                <div style='font-family:Syne,sans-serif;font-size:1.1rem;color:#34d399;font-weight:700;'>{tp:,}</div>
            </div>
            <div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.15);
                        border-radius:8px;padding:0.6rem;text-align:center;'>
                <div style='font-size:0.65rem;color:#475569;text-transform:uppercase;'>False Negative</div>
                <div style='font-family:Syne,sans-serif;font-size:1.1rem;color:#f87171;font-weight:700;'>{fn:,}</div>
            </div>
            <div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.15);
                        border-radius:8px;padding:0.6rem;text-align:center;'>
                <div style='font-size:0.65rem;color:#475569;text-transform:uppercase;'>False Positive</div>
                <div style='font-family:Syne,sans-serif;font-size:1.1rem;color:#fbbf24;font-weight:700;'>{fp:,}</div>
            </div>
            <div style='background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.15);
                        border-radius:8px;padding:0.6rem;text-align:center;'>
                <div style='font-size:0.65rem;color:#475569;text-transform:uppercase;'>True Negative</div>
                <div style='font-family:Syne,sans-serif;font-size:1.1rem;color:#60a5fa;font-weight:700;'>{tn:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    with col_b:
        st.markdown("<div class='section-header'><div class='section-dot' style='background:#8b5cf6;'></div><div class='section-title'>Performance vs Targets</div></div>", unsafe_allow_html=True)
        met_names  = ["Accuracy","Precision","Recall","F1-Score"]
        achieved   = [metrics["accuracy"]*100, metrics["precision"]*100,
                      metrics["recall"]*100, metrics["f1"]*100]
        targets    = [75.0, 90.0, 75.0, 80.0]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Target", x=met_names, y=targets,
            marker_color="rgba(51,65,85,0.5)",
            marker_line_color="rgba(99,102,241,0.3)",
            marker_line_width=1,
        ))
        fig_bar.add_trace(go.Bar(
            name="Achieved", x=met_names, y=achieved,
            marker_color=["#3b82f6","#8b5cf6","#10b981","#f59e0b"],
            text=[f"{v:.1f}%" for v in achieved],
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
        ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            height=260,
            barmode="group",
            yaxis=dict(range=[0,115], gridcolor="rgba(30,41,59,0.5)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(size=10)),
            bargap=0.3,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
 
        # Top features chart
        if model is not None and vectorizer is not None:
            try:
                feat_names = vectorizer.get_feature_names_out()
                coefs      = model.coef_[0]
                top_idx    = np.argsort(coefs)[-10:][::-1]
                top_words  = [feat_names[i] for i in top_idx]
                top_vals   = [float(coefs[i]) for i in top_idx]
 
                fig_feat = go.Figure(go.Bar(
                    x=top_vals[::-1], y=top_words[::-1],
                    orientation="h",
                    marker=dict(
                        color=top_vals[::-1],
                        colorscale=[[0,"#1d4ed8"],[0.5,"#7c3aed"],[1,"#ef4444"]],
                        showscale=False,
                    ),
                ))
                fig_feat.update_layout(
                    **PLOTLY_LAYOUT,
                    height=260,
                    title=dict(text="Top Bullying Indicators", font=dict(size=11, color="#64748b")),
                    xaxis=dict(gridcolor="rgba(30,41,59,0.5)"),
                    yaxis=dict(tickfont=dict(family="monospace", size=10)),
                )
                st.plotly_chart(fig_feat, use_container_width=True)
            except Exception:
                pass
 
    # Row 2: Dataset insights
    st.markdown("<div class='section-header' style='margin-top:0.5rem;'><div class='section-dot' style='background:#10b981;'></div><div class='section-title'>Dataset Insights</div></div>", unsafe_allow_html=True)
 
    col_c, col_d = st.columns(2)
 
    with col_c:
        if df is not None and label_col:
            try:
                dist   = df[label_col].value_counts()
                colors = ["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444","#6366f1"]
                fig_pie = go.Figure(go.Pie(
                    labels=dist.index.tolist(),
                    values=dist.values.tolist(),
                    hole=0.55,
                    marker=dict(colors=colors[:len(dist)],
                                line=dict(color="#070b14", width=2)),
                    textinfo="percent",
                    textfont=dict(size=10),
                ))
                fig_pie.update_layout(
                    **PLOTLY_LAYOUT,
                    height=260,
                    title=dict(text="Class Distribution", font=dict(size=11, color="#64748b")),
                    legend=dict(font=dict(size=9), orientation="v"),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            except Exception:
                st.info("Label column not found in dataset.")
        else:
            # Fallback static
            fig_pie = go.Figure(go.Pie(
                labels=["Religion","Age","Ethnicity","Gender","Not CB","Other CB"],
                values=[7995,7988,7955,7875,7657,7358],
                hole=0.55,
                marker=dict(colors=["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444","#6366f1"],
                            line=dict(color="#070b14", width=2)),
                textinfo="percent",
                textfont=dict(size=10),
            ))
            fig_pie.update_layout(
                **PLOTLY_LAYOUT, height=260,
                title=dict(text="Class Distribution (reference)", font=dict(size=11, color="#64748b")),
                legend=dict(font=dict(size=9)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
 
    with col_d:
        if df is not None and text_col:
            try:
                df_temp = df.copy()
                df_temp["text_len"] = df_temp[text_col].astype(str).str.split().str.len()
                fig_hist = px.histogram(
                    df_temp, x="text_len",
                    nbins=40,
                    color_discrete_sequence=["#3b82f6"],
                )
                fig_hist.update_traces(marker_line_color="#070b14", marker_line_width=0.5)
                fig_hist.update_layout(
                    **PLOTLY_LAYOUT,
                    height=260,
                    title=dict(text="Word Count Distribution", font=dict(size=11, color="#64748b")),
                    xaxis=dict(title="Words per text", gridcolor="rgba(30,41,59,0.5)"),
                    yaxis=dict(title="Frequency",      gridcolor="rgba(30,41,59,0.5)"),
                    bargap=0.05,
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception:
                st.info("Text length analysis unavailable.")
        else:
            st.markdown("""
            <div class='panel' style='height:260px;display:flex;align-items:center;
                                       justify-content:center;'>
                <div style='text-align:center;color:#334155;font-size:0.8rem;'>
                    Dataset not loaded — upload aggression_parsed_dataset.csv
                </div>
            </div>
            """, unsafe_allow_html=True)
 
    # Sample explorer
    if df is not None and text_col and label_col:
        st.markdown("<div class='section-header'><div class='section-dot' style='background:#f59e0b;'></div><div class='section-title'>Sample Explorer</div></div>", unsafe_allow_html=True)
        try:
            unique_labels = sorted(df[label_col].unique().tolist())
            sel_label = st.selectbox("Filter by label", ["All"] + [str(l) for l in unique_labels])
            df_show = df if sel_label == "All" else df[df[label_col].astype(str) == sel_label]
            sample = df_show[[text_col, label_col]].sample(min(8, len(df_show)), random_state=42)
            st.dataframe(
                sample.rename(columns={text_col:"Text", label_col:"Label"}),
                use_container_width=True,
                hide_index=True,
            )
        except Exception:
            pass
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SYSTEM INFO
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "System Info":
 
    st.markdown("""
    <div style='padding: 2rem 0 0.5rem;'>
        <div class='hero-eyebrow'>Diagnostics</div>
        <div style='font-family:Syne,sans-serif;font-size:1.75rem;font-weight:700;color:#f1f5f9;margin-bottom:0.5rem;'>
            System Information
        </div>
        <div style='font-size:0.875rem;color:#475569;'>
            Model configuration, file status, and deployment diagnostics.
        </div>
    </div>
    <hr class='fancy-divider'>
    """, unsafe_allow_html=True)
 
    col_l, col_r = st.columns(2)
 
    with col_l:
        st.markdown("""
        <div class='section-header'>
            <div class='section-dot'></div>
            <div class='section-title'>Model Configuration</div>
        </div>
        """, unsafe_allow_html=True)
 
        algo   = type(model).__name__ if model else "N/A"
        v_type = type(vectorizer).__name__ if vectorizer else "N/A"
        vocab  = len(vectorizer.vocabulary_) if vectorizer and hasattr(vectorizer, "vocabulary_") else "N/A"
        n_feat = vectorizer.max_features if vectorizer and hasattr(vectorizer, "max_features") else "N/A"
        ngrams = str(vectorizer.ngram_range) if vectorizer and hasattr(vectorizer, "ngram_range") else "N/A"
        coef   = model.coef_.shape if model and hasattr(model, "coef_") else "N/A"
        has_proba = "Yes" if model and hasattr(model, "predict_proba") else "No"
 
        rows = [
            ("Algorithm",     algo),
            ("Vectorizer",    v_type),
            ("Vocabulary size", str(vocab)),
            ("Max features",  str(n_feat)),
            ("N-gram range",  ngrams),
            ("Coef shape",    str(coef)),
            ("predict_proba", has_proba),
            ("NLTK available",str(NLTK_OK)),
        ]
 
        rows_html = "".join(f"""
        <tr>
            <td style='color:#475569;'>{k}</td>
            <td style='font-family:monospace;color:#93c5fd;'>{v}</td>
        </tr>
        """ for k, v in rows)
 
        st.markdown(f"""
        <div class='panel'>
            <table class='styled-table'>
                <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
 
        # Training info
        st.markdown("""
        <div class='section-header'>
            <div class='section-dot' style='background:#10b981;'></div>
            <div class='section-title'>Training Summary</div>
        </div>
        <div class='panel'>
            <div class='feature-item'>
                <div class='feature-icon' style='background:rgba(59,130,246,0.1);'>📚</div>
                <div class='feature-text'><strong>Training samples:</strong> 37,462 · <strong>Test samples:</strong> 9,366</div>
            </div>
            <div class='feature-item'>
                <div class='feature-icon' style='background:rgba(139,92,246,0.1);'>⚖️</div>
                <div class='feature-text'><strong>Class balance:</strong> SMOTE oversampling applied</div>
            </div>
            <div class='feature-item'>
                <div class='feature-icon' style='background:rgba(16,185,129,0.1);'>🎛️</div>
                <div class='feature-text'><strong>Tuning:</strong> GridSearchCV with stratified k-fold</div>
            </div>
            <div class='feature-item'>
                <div class='feature-icon' style='background:rgba(245,158,11,0.1);'>🗃️</div>
                <div class='feature-text'><strong>Dataset:</strong> Kaggle aggression_parsed_dataset</div>
            </div>
            <div class='feature-item' style='border-bottom:none;'>
                <div class='feature-icon' style='background:rgba(99,102,241,0.1);'>✂️</div>
                <div class='feature-text'><strong>Split:</strong> 80/20 stratified train/test</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    with col_r:
        st.markdown("""
        <div class='section-header'>
            <div class='section-dot' style='background:#8b5cf6;'></div>
            <div class='section-title'>File Status</div>
        </div>
        """, unsafe_allow_html=True)
 
        def fmt_size(path):
            if not os.path.exists(path):
                return None
            s = os.path.getsize(path)
            if s > 1_048_576:  return f"{s/1_048_576:.1f} MB"
            if s > 1_024:      return f"{s/1_024:.1f} KB"
            return f"{s} B"
 
        for key, fpath in FILE_MAP.items():
            exists = os.path.exists(fpath)
            size   = fmt_size(fpath)
            st.markdown(f"""
            <div class='file-row'>
                <span class='file-name'>{fpath}</span>
                <span>
                    {'<span class="file-ok">● Present · ' + size + '</span>' if exists
                     else '<span class="file-miss">● Missing</span>'}
                </span>
            </div>
            """, unsafe_allow_html=True)
 
        # Text preprocessing steps
        st.markdown("""
        <div class='section-header' style='margin-top:1.5rem;'>
            <div class='section-dot' style='background:#f59e0b;'></div>
            <div class='section-title'>Preprocessing Pipeline</div>
        </div>
        <div class='panel'>
        """, unsafe_allow_html=True)
 
        steps = [
            ("1", "Lowercase",           "Convert all text to lowercase"),
            ("2", "URL removal",         "Strip http/https and www URLs"),
            ("3", "Mention removal",     "Remove @user handles"),
            ("4", "Hashtag cleanup",     "Remove # symbol"),
            ("5", "Non-alpha removal",   "Keep only letters and spaces"),
            ("6", "Stopword filtering",  "Remove common English stopwords (NLTK)"),
            ("7", "Lemmatisation",       "Reduce words to root forms (WordNet)"),
            ("8", "TF-IDF transform",    "Vectorise: 5k features, unigram+bigram"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class='feature-item' style='padding:0.5rem 0;'>
                <div class='feature-icon' style='background:rgba(99,102,241,0.1);
                    font-size:0.65rem;font-weight:700;color:#818cf8;'>{num}</div>
                <div class='feature-text'><strong>{title}</strong> — {desc}</div>
            </div>
            """, unsafe_allow_html=True)
 
        st.markdown("</div>", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "About":
 
    st.markdown("""
    <div style='padding: 2rem 0 0.5rem;'>
        <div class='hero-eyebrow'>Background & Methodology</div>
        <div style='font-family:Syne,sans-serif;font-size:1.75rem;font-weight:700;color:#f1f5f9;margin-bottom:0.5rem;'>
            About this Project
        </div>
    </div>
    <hr class='fancy-divider'>
    """, unsafe_allow_html=True)
 
    col_l, col_r = st.columns([1.1, 1])
 
    with col_l:
        st.markdown("""
        <div class='panel'>
            <div class='panel-title'>Project Overview</div>
            <p style='font-size:0.875rem;color:#64748b;line-height:1.7;'>
                The <strong style='color:#94a3b8;'>Cyberbullying Detection System</strong> is an NLP-based
                machine learning pipeline designed to automatically detect and classify harmful text content
                from social media platforms.
            </p>
            <p style='font-size:0.875rem;color:#64748b;line-height:1.7;'>
                Built with <strong style='color:#94a3b8;'>scikit-learn</strong> and
                <strong style='color:#94a3b8;'>NLTK</strong>, the system processes raw social media text
                through a multi-stage cleaning and vectorisation pipeline before applying a tuned
                Logistic Regression classifier — achieving <strong style='color:#3b82f6;'>96.4% precision</strong>
                on held-out test data.
            </p>
            <p style='font-size:0.875rem;color:#64748b;line-height:1.7;'>
                Class imbalance is addressed with <strong style='color:#94a3b8;'>SMOTE oversampling</strong>,
                and hyperparameters are optimised using <strong style='color:#94a3b8;'>GridSearchCV</strong>
                to maximise generalisation across all 6 harm categories.
            </p>
        </div>
 
        <div class='panel' style='margin-top:0;'>
            <div class='panel-title'>Technical Stack</div>
            <div style='display:flex;flex-wrap:wrap;gap:8px;'>
                <span class='badge badge-blue'>Python 3.8+</span>
                <span class='badge badge-blue'>scikit-learn</span>
                <span class='badge badge-blue'>NLTK</span>
                <span class='badge badge-blue'>pandas / NumPy</span>
                <span class='badge badge-blue'>Streamlit</span>
                <span class='badge badge-blue'>Plotly</span>
                <span class='badge badge-blue'>imbalanced-learn (SMOTE)</span>
                <span class='badge badge-blue'>joblib</span>
                <span class='badge badge-blue'>scipy (sparse)</span>
                <span class='badge badge-green'>Logistic Regression</span>
                <span class='badge badge-green'>TF-IDF Vectorizer</span>
                <span class='badge badge-green'>GridSearchCV</span>
            </div>
        </div>
 
        <div class='panel' style='margin-top:0;'>
            <div class='panel-title'>References</div>
            <div style='font-size:0.8rem;color:#475569;line-height:1.7;'>
                <div style='margin-bottom:0.5rem;'>
                    📦 <strong style='color:#64748b;'>Dataset:</strong>
                    Kaggle — Cyberbullying Classification Dataset (Shah, S.)
                </div>
                <div style='margin-bottom:0.5rem;'>
                    📖 <strong style='color:#64748b;'>scikit-learn:</strong>
                    Pedregosa et al. (2011). JMLR 12, 2825–2830.
                </div>
                <div>
                    📖 <strong style='color:#64748b;'>SMOTE:</strong>
                    Chawla et al. (2002). JAIR 16, 321–357.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    with col_r:
        # Author card
        st.markdown("""
        <div class='panel' style='text-align:center;padding:2rem;'>
            <div style='width:72px;height:72px;border-radius:50%;
                        background:linear-gradient(135deg,#1d4ed8,#7c3aed);
                        display:flex;align-items:center;justify-content:center;
                        font-family:Syne,sans-serif;font-size:1.5rem;font-weight:700;
                        color:white;margin:0 auto 1rem;'>MH</div>
            <div style='font-family:Syne,sans-serif;font-size:1rem;
                        font-weight:600;color:#f1f5f9;'>Mir Shahadut Hossain</div>
            <div style='font-size:0.75rem;color:#475569;margin-top:4px;margin-bottom:1rem;'>
                ML Engineer · Data Scientist
            </div>
            <div style='display:flex;gap:8px;justify-content:center;flex-wrap:wrap;'>
                <span class='badge badge-blue'>NLP</span>
                <span class='badge badge-blue'>Machine Learning</span>
                <span class='badge badge-blue'>Python</span>
                <span class='badge badge-blue'>Data Science</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        # Ethical statement
        st.markdown("""
        <div class='panel' style='margin-top:0;border-color:rgba(245,158,11,0.2);'>
            <div class='panel-title' style='color:#fbbf24;'>⚖ Ethical Considerations</div>
            <div class='feature-item'>
                <div class='feature-icon' style='background:rgba(245,158,11,0.1);'>🔍</div>
                <div class='feature-text' style='font-size:0.8rem;'>
                    <strong>Support tool, not arbiter.</strong> Always pair automated flags with human review before taking action.
                </div>
            </div>
            <div class='feature-item'>
                <div class='feature-icon' style='background:rgba(245,158,11,0.1);'>⚡</div>
                <div class='feature-text' style='font-size:0.8rem;'>
                    <strong>Bias awareness.</strong> TF-IDF models can over-flag minority group discussions. Audit regularly.
                </div>
            </div>
            <div class='feature-item'>
                <div class='feature-icon' style='background:rgba(245,158,11,0.1);'>🔁</div>
                <div class='feature-text' style='font-size:0.8rem;'>
                    <strong>Vocabulary drift.</strong> Cyberbullying language evolves. Re-train periodically on fresh data.
                </div>
            </div>
            <div class='feature-item' style='border-bottom:none;'>
                <div class='feature-icon' style='background:rgba(245,158,11,0.1);'>🔒</div>
                <div class='feature-text' style='font-size:0.8rem;'>
                    <strong>Privacy first.</strong> Do not store user-submitted text beyond the current session.
                </div>
            </div>
        </div>
 
        <div class='panel' style='margin-top:0;'>
            <div class='panel-title'>Limitations</div>
            <div style='font-size:0.8rem;color:#475569;line-height:1.7;'>
                <div>• Context-blind — sarcasm and irony may be misclassified</div>
                <div>• English-only (NLTK stopwords/lemmatiser)</div>
                <div>• Static vocabulary — new slang not captured</div>
                <div>• Binary output — multi-class categories collapsed for inference</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border:none;border-top:1px solid rgba(30,41,59,0.5);margin-top:3rem;'>
<div style='display:flex;align-items:center;justify-content:space-between;
            padding:1rem 0;font-size:0.7rem;color:#334155;'>
    <div>
        <span style='font-family:Syne,sans-serif;font-weight:600;color:#475569;'>CyberGuard</span>
        · Cyberbullying Detection System v2.0
    </div>
    <div>Built with Streamlit · scikit-learn · NLTK</div>
</div>
""", unsafe_allow_html=True)

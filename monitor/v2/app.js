/**
 * Pulsar Monitor V2 - Main Application Controller.
 * Manages SSE stream with polling fallback, tab switching, notes, and canaries.
 */

import { renderHealthStrip, renderProgressStrip } from './health.js';
import { PulsarCharts } from './charts.js';

class PulsarApp {
  constructor() {
    this.charts = new PulsarCharts();
    this.isPaused = false;
    this.activeTab = 'canaries';
    this.canariesList = [];
    this.selectedCanary = 0;
    this.lastStepSeen = 0;
    this.sseSource = null;
  }

  init() {
    // 1. Initialize uPlot charts
    const containers = {
      learning: document.getElementById('chartLearning'),
      loops: document.getElementById('chartLoops'),
      optimization: document.getElementById('chartOptimization'),
      moe: document.getElementById('chartMoe'),
      dynamics: document.getElementById('chartDynamics'),
      system: document.getElementById('chartSystem'),
    };
    this.charts.init(containers);

    // 2. Setup theme toggle
    const themeBtn = document.getElementById('themeToggleBtn');
    if (themeBtn) {
      themeBtn.addEventListener('click', () => this.toggleTheme());
    }

    // 3. Setup pause button
    const pauseBtn = document.getElementById('pauseBtn');
    if (pauseBtn) {
      pauseBtn.addEventListener('click', () => {
        this.isPaused = !this.isPaused;
        pauseBtn.innerText = this.isPaused ? 'Resume' : 'Pause';
        pauseBtn.classList.toggle('active', this.isPaused);
      });
    }

    // 4. Setup tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const tab = e.target.getAttribute('data-tab');
        this.switchTab(tab);
      });
    });

    // 5. Setup notes form
    const noteBtn = document.getElementById('addNoteBtn');
    if (noteBtn) {
      noteBtn.addEventListener('click', () => this.submitNote());
    }

    // 6. Start data flow: SSE + periodic health & history
    this.initSSE();
    this.pollHealth();
    this.pollHistory();
    this.fetchCanaries();
    this.fetchCheckpoints();
    this.fetchNotes();

    setInterval(() => {
      if (!this.isPaused) {
        this.pollHealth();
        this.pollHistory();
      }
    }, 3000);

    setInterval(() => {
      if (!this.isPaused) {
        this.fetchCanaries();
        this.fetchCheckpoints();
      }
    }, 15000);
  }

  toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('pulsar_theme_v2', next);
    const btn = document.getElementById('themeToggleBtn');
    if (btn) btn.innerText = next === 'dark' ? '☀️ Light' : '🌙 Dark';
  }

  initSSE() {
    try {
      this.sseSource = new EventSource('/api/stream');
      this.sseSource.onmessage = (event) => {
        if (this.isPaused) return;
        try {
          const data = JSON.parse(event.data);
          renderProgressStrip(data);
          this.lastStepSeen = data.step || this.lastStepSeen;
        } catch (e) {
          // ignore parsing error
        }
      };
      this.sseSource.onerror = () => {
        // SSE disconnected; fallback to polling status
        this.sseSource.close();
        setInterval(() => {
          if (!this.isPaused) this.pollStatusFallback();
        }, 2000);
      };
    } catch (e) {
      setInterval(() => {
        if (!this.isPaused) this.pollStatusFallback();
      }, 2000);
    }
  }

  async pollStatusFallback() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        renderProgressStrip(data);
        this.lastStepSeen = data.step || this.lastStepSeen;
      }
    } catch (e) {}
  }

  async pollHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        renderHealthStrip(data);
      }
    } catch (e) {}
  }

  async pollHistory() {
    try {
      const res = await fetch('/api/history?max_points=2000');
      if (res.ok) {
        const data = await res.json();
        const records = data.history || [];
        this.charts.updateData(records);
      }
    } catch (e) {}
  }

  switchTab(tabId) {
    this.activeTab = tabId;
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b.getAttribute('data-tab') === tabId);
    });
    document.querySelectorAll('.tab-pane').forEach(p => {
      p.style.display = p.id === `tabPane_${tabId}` ? 'block' : 'none';
    });
  }

  async fetchCanaries() {
    try {
      const res = await fetch('/api/canaries');
      if (!res.ok) return;
      const data = await res.json();
      const milestones = data.milestones || [];
      if (!milestones.length) return;

      const latest = milestones[milestones.length - 1];
      this.canariesList = latest.canaries || [];

      const sidebar = document.getElementById('canarySidebar');
      if (sidebar) {
        sidebar.innerHTML = this.canariesList.map((c, i) => `
          <button class="canary-nav-item ${i === this.selectedCanary ? 'active' : ''}" onclick="window.pulsarApp.selectCanary(${i})">
            <strong>${c.category || c.tag || 'Prompt ' + (i+1)}</strong>
            <span style="font-size: 10px; color: var(--text-muted);">${c.tag || ''}</span>
          </button>
        `).join('');
      }
      this.renderCurrentCanary();
    } catch (e) {}
  }

  selectCanary(idx) {
    this.selectedCanary = idx;
    document.querySelectorAll('.canary-nav-item').forEach((b, i) => {
      b.classList.toggle('active', i === idx);
    });
    this.renderCurrentCanary();
  }

  renderCurrentCanary() {
    if (!this.canariesList[this.selectedCanary]) return;
    const c = this.canariesList[this.selectedCanary];
    const el = document.getElementById('canaryDisplayBody');
    if (el) {
      el.innerHTML = `
        <div style="color: var(--cyan); font-weight: 700; margin-bottom: 6px;">// EVALUATION PROMPT</div>
        <div style="margin-bottom: 14px;">${this.escapeHTML(c.prompt || '')}</div>
        <div style="color: var(--emerald); font-weight: 700; margin-bottom: 6px;">// MODEL COMPLETION</div>
        <div>${this.escapeHTML(c.completion || '')}</div>
      `;
    }
  }

  async fetchCheckpoints() {
    try {
      const res = await fetch('/api/checkpoints');
      if (!res.ok) return;
      const data = await res.json();
      const list = data.checkpoints || [];
      const tbody = document.getElementById('checkpointsTableBody');
      if (tbody) {
        tbody.innerHTML = list.map(c => `
          <tr>
            <td><strong>${c.filename}</strong></td>
            <td>${c.size_mb} MB</td>
            <td>${c.mtime_iso}</td>
            <td>
              ${c.is_best ? '<span class="status-pill-overall OK" style="font-size:9px;">BEST</span>' : ''}
              ${c.is_latest ? '<span class="status-pill-overall WATCH" style="font-size:9px;">LATEST</span>' : ''}
              ${c.is_milestone ? '<span class="brand-v2-tag">MILESTONE</span>' : ''}
            </td>
          </tr>
        `).join('');
      }
    } catch (e) {}
  }

  async fetchNotes() {
    try {
      const res = await fetch('/api/notes');
      if (!res.ok) return;
      const data = await res.json();
      const notes = data.notes || [];
      const listEl = document.getElementById('notesListContainer');
      if (listEl) {
        if (!notes.length) {
          listEl.innerHTML = '<div style="color: var(--text-muted); font-size: 12px;">No annotations recorded yet. Use the composer above to pin a note.</div>';
          return;
        }
        listEl.innerHTML = notes.slice().reverse().map(n => `
          <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 10px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); margin-bottom: 4px;">
              <span><strong>Step ${n.step.toLocaleString()}</strong> • ${n.author}</span>
              <span>${n.created_at}</span>
            </div>
            <div style="color: var(--text-primary); font-size: 12px;">${this.escapeHTML(n.text)}</div>
          </div>
        `).join('');
      }
    } catch (e) {}
  }

  async submitNote() {
    const input = document.getElementById('noteTextInput');
    const stepInput = document.getElementById('noteStepInput');
    if (!input || !input.value.trim()) return;

    const text = input.value.trim();
    const step = parseInt(stepInput?.value || this.lastStepSeen || 0, 10);

    try {
      const res = await fetch('/api/notes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ step, text, author: 'User' })
      });
      if (res.ok) {
        input.value = '';
        this.fetchNotes();
      }
    } catch (e) {}
  }

  escapeHTML(str) {
    return str.replace(/[&<>'"]/g, tag => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    }[tag] || tag));
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.pulsarApp = new PulsarApp();
  window.pulsarApp.init();
});

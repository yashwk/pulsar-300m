/**
 * Pulsar Monitor V2 - uPlot 6-Chart Engine.
 * Sub-15ms canvas renders with synchronized crosshairs.
 */

// uPlot cursor sync group
const cursorSync = uPlot.sync("pulsar_v2");

export class PulsarCharts {
  constructor() {
    this.charts = {};
  }

  init(containers) {
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.07)' : 'rgba(0, 0, 0, 0.07)';
    const textColor = isDark ? '#94a3b8' : '#64748b';

    const baseAxes = [
      {
        grid: { stroke: gridColor, width: 1 },
        ticks: { stroke: gridColor, width: 1 },
        stroke: textColor,
        font: "11px 'JetBrains Mono'",
        size: 24,
      },
      {
        grid: { stroke: gridColor, width: 1 },
        ticks: { stroke: gridColor, width: 1 },
        stroke: textColor,
        font: "11px 'JetBrains Mono'",
        size: 42,
      }
    ];

    // 1. Chart 1: Learning (Final-Pulse Train CE vs Val CE Final)
    if (containers.learning) {
      const opts1 = {
        width: containers.learning.clientWidth || 500,
        height: 220,
        cursor: { sync: cursorSync },
        scales: { x: { time: false } },
        axes: baseAxes,
        series: [
          { label: "Step" },
          { label: "Train Final CE", stroke: "#38bdf8", width: 1.5 },
          { label: "Val CE Final", stroke: "#10b981", width: 2, points: { show: true, size: 5 } },
          { label: "Weighted Train Loss", stroke: "#64748b80", width: 1, dash: [3, 3] },
        ]
      };
      this.charts.learning = new uPlot(opts1, [[], [], [], []], containers.learning);
    }

    // 2. Chart 2: Recurrent Loops (Per-Pulse Val CE & Latent Gain)
    if (containers.loops) {
      const opts2 = {
        width: containers.loops.clientWidth || 500,
        height: 220,
        cursor: { sync: cursorSync },
        scales: { x: { time: false } },
        axes: baseAxes,
        series: [
          { label: "Step" },
          { label: "Pulse 1 CE", stroke: "#f59e0b", width: 1.2, dash: [3, 3] },
          { label: "Pulse 2 CE", stroke: "#06b6d4", width: 1.2, dash: [3, 3] },
          { label: "Pulse 3 CE", stroke: "#10b981", width: 2 },
          { label: "Latent Gain (ΔL)", stroke: "#a855f7", width: 1.5 },
        ]
      };
      this.charts.loops = new uPlot(opts2, [[], [], [], [], []], containers.loops);
    }

    // 3. Chart 3: Optimization & Gradients
    if (containers.optimization) {
      const opts3 = {
        width: containers.optimization.clientWidth || 500,
        height: 220,
        cursor: { sync: cursorSync },
        scales: { x: { time: false } },
        axes: baseAxes,
        series: [
          { label: "Step" },
          { label: "Grad Norm Total", stroke: "#f8fafc", width: 1.5 },
          { label: "Muon Norm", stroke: "#10b981", width: 1.2 },
          { label: "Adam Norm", stroke: "#94a3b8", width: 1.2 },
          { label: "Clip Limit (1.0)", stroke: "#f43f5e", width: 1, dash: [4, 4] },
        ]
      };
      this.charts.optimization = new uPlot(opts3, [[], [], [], [], []], containers.optimization);
    }

    // 4. Chart 4: MoE Routing & Drops
    if (containers.moe) {
      const opts4 = {
        width: containers.moe.clientWidth || 500,
        height: 220,
        cursor: { sync: cursorSync },
        scales: { x: { time: false } },
        axes: baseAxes,
        series: [
          { label: "Step" },
          { label: "Drop Rate %", stroke: "#f43f5e", width: 1.5 },
          { label: "Router Confidence", stroke: "#38bdf8", width: 1.5 },
          { label: "Worst Expert Share %", stroke: "#f59e0b", width: 1.2 },
        ]
      };
      this.charts.moe = new uPlot(opts4, [[], [], [], []], containers.moe);
    }

    // 5. Chart 5: Recurrent State Dynamics (Δh Contraction)
    if (containers.dynamics) {
      const opts5 = {
        width: containers.dynamics.clientWidth || 500,
        height: 220,
        cursor: { sync: cursorSync },
        scales: { x: { time: false } },
        axes: baseAxes,
        series: [
          { label: "Step" },
          { label: "Δh1 (Pulse 1)", stroke: "#f59e0b", width: 1.5 },
          { label: "Δh2 (Pulse 2)", stroke: "#06b6d4", width: 1.5 },
          { label: "Δh3 (Pulse 3)", stroke: "#10b981", width: 1.5 },
        ]
      };
      this.charts.dynamics = new uPlot(opts5, [[], [], [], []], containers.dynamics);
    }

    // 6. Chart 6: System & Hardware
    if (containers.system) {
      const opts6 = {
        width: containers.system.clientWidth || 500,
        height: 220,
        cursor: { sync: cursorSync },
        scales: { x: { time: false } },
        axes: baseAxes,
        series: [
          { label: "Step" },
          { label: "Throughput (tok/s)", stroke: "#10b981", width: 1.5 },
          { label: "Peak VRAM (GB)", stroke: "#a855f7", width: 1.5 },
          { label: "Hardware MFU %", stroke: "#38bdf8", width: 1.2 },
        ]
      };
      this.charts.system = new uPlot(opts6, [[], [], [], []], containers.system);
    }

    // Auto resize listener
    window.addEventListener("resize", () => {
      for (const [key, chart] of Object.entries(this.charts)) {
        const parent = containers[key];
        if (parent && chart) {
          chart.setSize({ width: parent.clientWidth || 500, height: 220 });
        }
      }
    });
  }

  updateData(records) {
    if (!records || !records.length) return;

    const steps = [];
    const trFinal = [];
    const valFinal = [];
    const trWeighted = [];

    const p1Loss = [];
    const p2Loss = [];
    const p3Loss = [];
    const latentGain = [];

    const gradTotal = [];
    const gradMuon = [];
    const gradAdam = [];
    const clipLimit = [];

    const dropRate = [];
    const routerConf = [];
    const worstShare = [];

    const dh1 = [];
    const dh2 = [];
    const dh3 = [];

    const tps = [];
    const vram = [];
    const mfu = [];

    for (const r of records) {
      const s = r.step || 0;
      steps.push(s);

      // Learning
      trFinal.push(r.train_ce_final !== undefined ? r.train_ce_final : r.train_loss);
      valFinal.push(r.val_ce_final !== undefined ? r.val_ce_final : null);
      trWeighted.push(r.train_loss !== undefined ? r.train_loss : null);

      // Loops
      const pl = r.pulse_losses;
      if (Array.isArray(pl) && pl.length >= 3) {
        p1Loss.push(pl[0]); p2Loss.push(pl[1]); p3Loss.push(pl[2]);
      } else {
        p1Loss.push(null); p2Loss.push(null); p3Loss.push(null);
      }
      latentGain.push(r.latent_gain !== undefined ? r.latent_gain : null);

      // Gradients
      gradTotal.push(r.grad_norm_total !== undefined ? r.grad_norm_total : null);
      gradMuon.push(r.grad_norm_muon !== undefined ? r.grad_norm_muon : null);
      gradAdam.push(r.grad_norm_adam !== undefined ? r.grad_norm_adam : null);
      clipLimit.push(1.0);

      // MoE
      dropRate.push(r.moe_drop_rate !== undefined ? r.moe_drop_rate * 100 : null);
      routerConf.push(r.router_confidence !== undefined ? r.router_confidence : null);
      worstShare.push(r.worst_expert_share_pct !== undefined ? r.worst_expert_share_pct : (Array.isArray(r.expert_pcts) ? Math.max(...r.expert_pcts) : null));

      // Contraction Dynamics
      const pd = r.pulse_deltas;
      if (Array.isArray(pd) && pd.length >= 3) {
        dh1.push(pd[0]); dh2.push(pd[1]); dh3.push(pd[2]);
      } else {
        dh1.push(null); dh2.push(null); dh3.push(null);
      }

      // Hardware
      tps.push(r.tok_per_sec !== undefined ? r.tok_per_sec : null);
      vram.push(r.vram_peak !== undefined ? r.vram_peak : null);
      mfu.push(r.mfu_pct !== undefined ? r.mfu_pct : null);
    }

    if (this.charts.learning) {
      this.charts.learning.setData([steps, trFinal, valFinal, trWeighted]);
    }
    if (this.charts.loops) {
      this.charts.loops.setData([steps, p1Loss, p2Loss, p3Loss, latentGain]);
    }
    if (this.charts.optimization) {
      this.charts.optimization.setData([steps, gradTotal, gradMuon, gradAdam, clipLimit]);
    }
    if (this.charts.moe) {
      this.charts.moe.setData([steps, dropRate, routerConf, worstShare]);
    }
    if (this.charts.dynamics) {
      this.charts.dynamics.setData([steps, dh1, dh2, dh3]);
    }
    if (this.charts.system) {
      this.charts.system.setData([steps, tps, vram, mfu]);
    }
  }
}

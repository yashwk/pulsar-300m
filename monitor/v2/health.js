/**
 * Pulsar Monitor V2 - Tier 1 Health Strip Controller.
 * Evaluates and renders real-time checks from /api/health and /api/status.
 */

export function renderHealthStrip(healthData) {
  if (!healthData) return;

  // 1. Overall Status Pill
  const overallBadge = document.getElementById('overallStatusPill');
  const overallText = document.getElementById('overallStatusText');
  const status = healthData.status || 'OK';

  if (overallBadge) {
    overallBadge.className = `status-pill-overall ${status}`;
  }
  if (overallText) {
    overallText.innerText = status === 'ALERT' ? 'ALERT - INTERVENE' : (status === 'WATCH' ? 'WATCH - CAUTION' : 'HEALTHY - OK');
  }

  // 2. 9 Detailed Checks
  const checks = healthData.checks || {};
  renderCheckCard('checkRunState', checks.run_state, checks.run_state?.value || 'running', checks.run_state?.threshold);
  renderCheckCard('checkProgress', checks.progress, checks.progress?.value || '--', checks.progress?.eta ? `ETA: ${checks.progress.eta}` : '--');
  
  const lossCheck = checks.loss || {};
  const lossVal = (lossCheck.train_ce_final !== undefined && lossCheck.val_ce_final !== undefined)
    ? `Tr: ${lossCheck.train_ce_final} | Val: ${lossCheck.val_ce_final}`
    : (lossCheck.train_ce_final ? `Tr: ${lossCheck.train_ce_final}` : '--');
  const lossSub = lossCheck.loss_gap !== undefined ? `Gap: ${lossCheck.loss_gap} nats` : '--';
  renderCheckCard('checkLoss', lossCheck, lossVal, lossSub);

  const gradCheck = checks.gradients || {};
  const gradVal = gradCheck.grad_norm_total !== undefined ? `Norm: ${gradCheck.grad_norm_total}` : '--';
  const gradSub = gradCheck.clip_fraction !== undefined ? `Clip: ${(gradCheck.clip_fraction * 100).toFixed(1)}%` : '--';
  renderCheckCard('checkGradients', gradCheck, gradVal, gradSub);

  const tpsCheck = checks.throughput || {};
  const tpsVal = tpsCheck.tok_per_sec !== undefined ? `${(tpsCheck.tok_per_sec / 1000).toFixed(1)}k tok/s` : '--';
  const tpsSub = tpsCheck.vram_peak !== undefined ? `${tpsCheck.vram_peak} GB (${tpsCheck.mfu_pct || 0}% MFU)` : '--';
  renderCheckCard('checkThroughput', tpsCheck, tpsVal, tpsSub);

  const dropsCheck = checks.moe_drops || {};
  const dropsVal = dropsCheck.drop_rate_pct !== undefined ? `${dropsCheck.drop_rate_pct}% Drops` : '--';
  const dropsSub = dropsCheck.capacity ? `Cap: ${dropsCheck.capacity} slots` : '--';
  renderCheckCard('checkDrops', dropsCheck, dropsVal, dropsSub);

  const balCheck = checks.moe_balance || {};
  const balVal = balCheck.worst_expert_share_pct !== undefined ? `Worst: ${balCheck.worst_expert_share_pct}%` : '--';
  const balSub = balCheck.router_confidence !== undefined ? `Conf: ${balCheck.router_confidence} (B: ${balCheck.max_bias})` : '--';
  renderCheckCard('checkBalance', balCheck, balVal, balSub);

  const loopCheck = checks.recurrent_loops || {};
  const loopVal = loopCheck.latent_gain !== undefined ? `Gain: +${loopCheck.latent_gain}` : '--';
  renderCheckCard('checkLoops', loopCheck, loopVal, 'Recurrent Gain (P1 → P3)');

  const numCheck = checks.numerics || {};
  const numVal = numCheck.nan_inf_found ? 'NaN / Inf DETECTED' : '0 NaN / Inf';
  renderCheckCard('checkNumerics', numCheck, numVal, 'Loss & Grads Stable');
}

function renderCheckCard(elemId, checkObj, valText, subText) {
  const el = document.getElementById(elemId);
  if (!el) return;

  const status = checkObj?.status || 'OK';
  el.className = `health-check-card ${status}`;

  const valEl = el.querySelector('.check-value');
  if (valEl) valEl.innerText = valText !== undefined && valText !== null ? valText : 'n/a';

  const subEl = el.querySelector('.check-sub');
  if (subEl) subEl.innerText = subText !== undefined && subText !== null ? subText : 'n/a';

  const tipEl = el.querySelector('.tooltip-content');
  if (tipEl) {
    const desc = checkObj?.description || 'Metric status check.';
    const th = checkObj?.threshold ? `<br><strong>Threshold:</strong> ${checkObj.threshold}` : '';
    tipEl.innerHTML = `${desc}${th}`;
  }
}

export function renderProgressStrip(statusData) {
  if (!statusData) return;

  const curStep = statusData.step || 0;
  const targetStep = statusData.active_target || statusData.target_step || statusData.max_steps || 3500;
  const pct = targetStep > 0 ? Math.min(100, (curStep / targetStep) * 100).toFixed(1) : '0.0';

  const barEl = document.getElementById('sessionProgressBarV2');
  if (barEl) barEl.style.width = `${pct}%`;

  const stepLabel = document.getElementById('stepProgressLabel');
  if (stepLabel) stepLabel.innerText = `Step ${curStep.toLocaleString()} / ${targetStep.toLocaleString()} (${pct}%)`;

  const tokensSeen = statusData.tokens_seen || 0;
  const datasetTokens = statusData.train_dataset_tokens || 0;
  const trainShards = Array.isArray(statusData.train_shards) ? statusData.train_shards : [];
  const numShards = trainShards.length > 0 ? trainShards.length : (statusData.num_train_shards || 1);

  const horizonEl = document.getElementById('shardHorizonLabel');
  if (horizonEl) {
    if (trainShards.length > 0) {
      const labels = trainShards.map(s => {
        const m = s.match(/shard(\d+)/i);
        return m ? `Shard ${m[1]}` : s.replace(/\.bin$/, '');
      });
      const shardStr = labels.length === 1 ? labels[0] : (labels.length === 2 ? `${labels[0]} & ${labels[1]}` : `${labels.slice(0, -1).join(', ')} & ${labels[labels.length - 1]}`);
      horizonEl.innerText = `${shardStr} Active (${(tokensSeen / 1e6).toFixed(1)}M / ${(datasetTokens / 1e6).toFixed(1)}M tok)`;
    } else {
      horizonEl.innerText = `${numShards} Shards Active (${(tokensSeen / 1e6).toFixed(1)}M / ${(datasetTokens / 1e6).toFixed(1)}M tok)`;
    }
  }

  const phaseEl = document.getElementById('runPhaseTag');
  if (phaseEl) phaseEl.innerText = statusData.phase || 'Stable Phase';
}

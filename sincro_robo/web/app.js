const dom = (id) => document.getElementById(id);
const app = { state: null, config: null, sessions: [], busy: false };

function fmt(value, digits = 2) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function toast(message, kind = "ok") {
  const item = document.createElement("div");
  item.className = `toast ${kind === "error" ? "error" : ""}`;
  item.textContent = message;
  dom("toast-region").appendChild(item);
  setTimeout(() => item.remove(), 4400);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    let message = `Falha HTTP ${response.status}`;
    try { message = (await response.json()).detail || message; } catch (_) { /* no-op */ }
    throw new Error(message);
  }
  return response.json();
}

function setService(id, service) {
  const node = dom(id);
  const status = service?.status || "offline";
  node.className = `service-status ${status}`;
  node.querySelector("b").textContent = status === "online" ? "online" : status === "error" ? "falha" : status;
  node.title = service?.error || "";
}

function renderVision(vision) {
  const status = dom("vision-state");
  const hasVision = Boolean(vision);
  if (!hasVision) {
    status.className = "vision-state invalid";
    status.innerHTML = "<i></i> Sem instância válida";
  } else if (vision.stable) {
    status.className = "vision-state stable";
    status.innerHTML = "<i></i> Visão estável";
  } else {
    status.className = "vision-state waiting";
    status.innerHTML = "<i></i> Estabilizando";
  }
  dom("vision-x").textContent = hasVision ? fmt(vision.x, 1) : "—";
  dom("vision-y").textContent = hasVision ? fmt(vision.y, 1) : "—";
  dom("vision-angle").textContent = hasVision ? fmt(vision.angle_deg, 1) : "—";
  dom("vision-confidence").textContent = hasVision ? `${fmt(vision.confidence * 100, 1)}%` : "—";
  dom("axis-quality").textContent = hasVision ? fmt(vision.axis_quality, 2) : "—";
  dom("sigma-x").textContent = hasVision ? fmt(vision.sigma_x, 2) : "—";
  dom("sigma-y").textContent = hasVision ? fmt(vision.sigma_y, 2) : "—";
  dom("sigma-angle").textContent = hasVision ? fmt(vision.sigma_angle_deg, 2) : "—";
  dom("frame-resolution").textContent = hasVision ? `${vision.frame_width} × ${vision.frame_height} px` : "— × — px";
  dom("confidence-bar").style.width = hasVision ? `${Math.min(100, vision.confidence * 100)}%` : "0";
  dom("axis-bar").style.width = hasVision ? `${Math.min(100, vision.axis_quality * 18)}%` : "0";
}

function renderRobot(robot) {
  ["x", "y", "z", "rx", "ry", "rz"].forEach((key) => {
    dom(`robot-${key}`).textContent = robot ? fmt(robot[key], key.startsWith("r") ? 2 : 2) : "—";
  });
  dom("pose-freshness").textContent = robot ? "LEITURA RECENTE" : "SEM LEITURA";
}

function renderGates(state) {
  const gates = state.capture_readiness || {};
  const step = gates.step || "vision";
  const frozen = state.frozen_vision;
  document.querySelectorAll("#gate-list [data-gate]").forEach((node) => {
    node.classList.toggle("ok", Boolean(gates[node.dataset.gate]));
  });
  const canCapture = Boolean(gates.session && gates.plan && gates.ready) && !app.busy;
  dom("capture-button").disabled = !canCapture;
  const discard = dom("discard-button");
  discard.hidden = step !== "robot";
  discard.disabled = app.busy || step !== "robot";
  if (step === "robot") {
    dom("capture-label").textContent = "CAPTURAR COORDENADAS DO ROBÔ (2/2)";
    dom("capture-sublabel").textContent = "Congelar pose CIP e gravar o ponto";
  } else {
    dom("capture-label").textContent = "CAPTURAR COORDENADAS DA VISÃO (1/2)";
    dom("capture-sublabel").textContent = "Congelar Xv, Yv e θ";
  }
  const frozenBox = dom("frozen-vision");
  if (frozen) {
    frozenBox.hidden = false;
    const vision = frozen.vision;
    dom("frozen-x").textContent = fmt(vision.x, 1);
    dom("frozen-y").textContent = fmt(vision.y, 1);
    dom("frozen-angle").textContent = `${fmt(vision.angle_deg, 1)}°`;
  } else {
    frozenBox.hidden = true;
  }
  dom("capture-help").textContent = !state.active_session_id
    ? "Ative uma sessão e um plano para começar."
    : state.active_plan_z === null
      ? "Selecione o plano Z em que a coleta será feita."
      : step === "robot"
        ? "Visão congelada. Posicione o robô e capture a pose. A câmera pode perder o objeto."
        : "Os indicadores são só status. Capture a visão quando o molde estiver visível.";
}

function renderPlans(session) {
  const container = dom("plans-list");
  if (!session) {
    container.innerHTML = '<div class="empty-inline">Crie ou abra uma sessão para visualizar os planos.</div>';
    return;
  }
  container.innerHTML = session.plans.map((plan) => {
    const dots = Array.from({ length: plan.target }, (_, index) => {
      const done = index < plan.count ? "done" : "";
      const role = index < 5 ? "adjustment" : index < 7 ? "validation" : "expansion";
      return `<i class="${role} ${done}"></i>`;
    }).join("");
    const evaluation = plan.evaluation;
    const metricText = !evaluation.ready ? "Aguardando pontos de ajuste" : evaluation.metrics.passed ? "Dentro dos limites provisórios" : "Requer atenção / expansão";
    const metricClass = evaluation.ready ? (evaluation.metrics.passed ? "pass" : "fail") : "";
    return `<button class="plan-card ${session.active_plan_z === plan.z ? "active" : ""}" data-plan="${plan.z}">
      <div class="plan-card-head"><strong>Z = ${fmt(plan.z, 0)} mm</strong><span>${plan.count} / ${plan.target}</span></div>
      <div class="point-dots">${dots}</div><small class="${metricClass}">${metricText}</small>
    </button>`;
  }).join("");
  container.querySelectorAll("[data-plan]").forEach((button) => button.addEventListener("click", () => activatePlan(Number(button.dataset.plan))));
}

function currentPlan(session) {
  return session?.plans?.find((plan) => plan.z === session.active_plan_z) || null;
}

function renderCaptureTarget(session) {
  const plan = currentPlan(session);
  dom("current-plane").textContent = plan ? `Z ${fmt(plan.z, 0)} mm` : "Z —";
  dom("next-point").textContent = plan?.suggestion ? `Ponto ${String(plan.suggestion.index).padStart(2, "0")} / ${plan.target}` : plan?.complete ? "Plano concluído" : "Selecione um plano";
  dom("region-name").textContent = plan?.suggestion?.region || "—";
  dom("point-role").textContent = plan?.suggestion ? (plan.suggestion.role === "validation" ? "ponto de validação" : "ponto de ajuste") : "—";
  const [x, y] = plan?.suggestion?.normalized || [0.5, 0.5];
  dom("target-dot").style.left = `${x * 100}%`;
  dom("target-dot").style.top = `${y * 100}%`;
}

function renderFeedback(feedback) {
  const box = dom("capture-feedback");
  if (!feedback) {
    box.hidden = true;
    box.className = "capture-feedback";
    return;
  }
  box.hidden = false;
  box.className = `capture-feedback ${feedback.suggestion_kind || ""}`;
  dom("feedback-rmse").textContent = feedback.message || "—";
  dom("feedback-suggestion").textContent = `Sugestão: ${feedback.suggestion || "—"}`;
}

function renderMetrics(session) {
  const plan = currentPlan(session);
  const metrics = plan?.evaluation?.ready ? plan.evaluation.metrics : null;
  dom("metric-xy-rms").textContent = metrics ? `${fmt(metrics.xy_mm.rms)} mm` : "—";
  dom("metric-xy-p95").textContent = metrics ? `${fmt(metrics.xy_mm.p95)} mm` : "—";
  dom("metric-angle-rms").textContent = metrics ? `${fmt(metrics.angle_deg.rms)}°` : "—";
  dom("metric-max").textContent = metrics ? `${fmt(metrics.xy_mm.max)} mm` : "—";
  dom("metric-status").textContent = metrics ? (metrics.passed ? "dentro dos limites" : "fora do limite provisório") : "aguardando modelo";
}

function renderPairs(session) {
  const rows = session?.pairs || [];
  dom("pairs-count").textContent = `${rows.length} ${rows.length === 1 ? "ponto" : "pontos"}`;
  dom("pairs-table").innerHTML = rows.length ? rows.map((pair) => `<tr>
    <td>${String(pair.point_index).padStart(2, "0")}</td><td>Z ${fmt(pair.plan_z, 0)}</td>
    <td><span class="tag ${pair.role === "validation" ? "validation" : ""}">${pair.role === "validation" ? "validação" : "ajuste"}</span></td>
    <td>${fmt(pair.vision.x, 1)} · ${fmt(pair.vision.y, 1)}</td><td>${fmt(pair.vision.angle_deg, 1)}°</td>
    <td>${fmt(pair.robot.x, 1)} · ${fmt(pair.robot.y, 1)} · ${fmt(pair.robot.z, 1)}</td><td>${fmt(pair.robot.rz, 1)}°</td>
    <td>${pair.residual_xy === null ? "—" : `${fmt(pair.residual_xy)} mm`}</td>
    <td><span class="tag ${pair.status === "suspect" ? "suspect" : ""}">${pair.status === "suspect" ? "suspeito" : "aceito"}</span></td>
  </tr>`).join("") : '<tr><td colspan="9" class="empty-table">Nenhum ponto gravado nesta sessão.</td></tr>';
}

function renderSession(session) {
  dom("session-name").textContent = session?.name || "Nenhuma sessão selecionada";
  dom("session-state").textContent = session ? session.status.toUpperCase() : "AGUARDANDO";
  dom("export-button").disabled = !session;
  renderPlans(session);
  renderCaptureTarget(session);
  renderMetrics(session);
  renderPairs(session);
  renderFeedback(app.state?.capture_feedback);
}

function render(state) {
  setService("status-camera", state.hardware.camera);
  setService("status-model", state.hardware.model);
  setService("status-plc", state.hardware.plc);
  renderVision(state.vision);
  renderRobot(state.robot);
  renderGates(state);
  renderSession(state.session);
}

async function refreshState() {
  try {
    app.state = await api("/api/state");
    render(app.state);
  } catch (error) {
    setService("status-camera", { status: "error", error: error.message });
  }
}

async function refreshSessions(selected = null) {
  app.sessions = await api("/api/sessions");
  const select = dom("session-select");
  const selectedId = selected || app.state?.active_session_id || select.value;
  select.innerHTML = '<option value="">Selecionar sessão…</option>' + app.sessions.map((session) => `<option value="${session.id}">${session.name}</option>`).join("");
  select.value = selectedId || "";
}

async function activatePlan(plan) {
  if (!app.state?.active_session_id) return;
  try {
    await api(`/api/sessions/${app.state.active_session_id}/plans/${plan}/activate`, { method: "POST" });
    await refreshState();
    toast(`Plano Z=${fmt(plan, 0)} mm ativado.`);
  } catch (error) { toast(error.message, "error"); }
}

async function capture() {
  if (app.busy) return;
  app.busy = true;
  dom("capture-button").disabled = true;
  try {
    const result = await api("/api/capture", { method: "POST" });
    await refreshState();
    if (result.saved) {
      toast("Ponto registrado.");
    }
  } catch (error) { toast(error.message, "error"); }
  finally { app.busy = false; }
}

async function discardFrozen() {
  if (app.busy) return;
  try {
    await api("/api/candidate/decision", { method: "POST", body: JSON.stringify({ confirm: false }) });
    await refreshState();
    toast("Visão congelada descartada; nenhum ponto foi gravado.");
  } catch (error) { toast(error.message, "error"); }
}

async function initialize() {
  setInterval(() => { dom("clock").textContent = new Date().toLocaleString("pt-BR", { hour12: false }); }, 500);
  try {
    app.config = await api("/api/config");
    dom("version-label").textContent = `v${app.config.version} · ${app.config.git_revision}`;
    const values = Object.values(app.config.mode);
    const simulated = values.every((value) => value === "synthetic");
    dom("mode-description").textContent = simulated ? "Câmera, modelo e CLP simulados — nenhum hardware será acionado" : `Câmera ${app.config.mode.camera} · modelo ${app.config.mode.model} · CLP ${app.config.mode.plc}`;
    await refreshSessions();
    await refreshState();
    setInterval(refreshState, 700);
    setInterval(() => {
      const frame = dom("camera-frame");
      frame.src = `/api/frame?t=${Date.now()}`;
    }, 280);
  } catch (error) { toast(error.message, "error"); }
}

dom("camera-frame").addEventListener("load", () => {
  dom("camera-frame").classList.add("visible");
  dom("camera-empty").classList.add("hidden");
});
dom("new-session").addEventListener("click", () => dom("session-dialog").showModal());
dom("activate-session").addEventListener("click", async () => {
  const id = dom("session-select").value;
  if (!id) return toast("Selecione uma sessão.", "error");
  try { await api(`/api/sessions/${id}/activate`, { method: "POST" }); await refreshState(); toast("Sessão aberta."); }
  catch (error) { toast(error.message, "error"); }
});
dom("session-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = dom("session-input-name").value.trim();
  const planes = dom("session-input-planes").value.split(",").map((value) => Number(value.trim())).filter(Number.isFinite);
  try {
    const session = await api("/api/sessions", { method: "POST", body: JSON.stringify({ name, planes }) });
    await api(`/api/sessions/${session.id}/activate`, { method: "POST" });
    dom("session-dialog").close();
    dom("session-form").reset();
    dom("session-input-planes").value = "0, 200, 400";
    await refreshSessions(session.id);
    await refreshState();
    toast("Sessão criada. Selecione o primeiro plano Z.");
  } catch (error) { toast(error.message, "error"); }
});
dom("capture-button").addEventListener("click", capture);
dom("discard-button").addEventListener("click", discardFrozen);
dom("export-button").addEventListener("click", async () => {
  if (!app.state?.active_session_id) return;
  try {
    const files = await api(`/api/sessions/${app.state.active_session_id}/export`, { method: "POST" });
    Object.values(files).forEach((url) => window.open(url, "_blank", "noopener"));
    toast("JSON de calibração e CSV de pontos exportados.");
  } catch (error) { toast(error.message, "error"); }
});

initialize();

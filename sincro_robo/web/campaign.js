let campaignSignature = "";
let campaignBusy = false;

function campaignPost(path, body) {
  campaignBusy = true;
  const options = { method: "POST" };
  if (body !== undefined) options.body = JSON.stringify(body);
  return api(path, options)
    .then(async () => {
      campaignSignature = "";
      await refreshState();
    })
    .catch((error) => toast(error.message, "error"))
    .finally(() => {
      campaignBusy = false;
    });
}

function renderCampaign(campaign) {
  const body = document.getElementById("campaign-body");
  if (!body || campaignBusy) return;
  if (!campaign) {
    campaignSignature = "none";
    body.innerHTML = '<p class="campaign-lead">Nenhuma campanha aberta. A coleta do molde abaixo continua disponível.</p>';
    return;
  }
  const live = document.getElementById("campaign-live");
  const liveText = campaign.detection?.found
    ? `Tabuleiro visto · ${campaign.detection.count} cantos`
    : "Tabuleiro não detectado neste frame";
  if (live) live.textContent = liveText;
  paintFocus(campaign);
  const signature = JSON.stringify({
    stage: campaign.stage,
    gate: campaign.gate,
    accepted: campaign.accepted_count,
    coverage: campaign.coverage,
    checks: campaign.checks,
    focusStable: campaign.focus?.stable,
    focusDrift: campaign.focus?.drift,
    focusLocked: campaign.focus?.locked_peak,
    reprojection: campaign.reprojection,
    frame: campaign.frame_points,
    direction: campaign.direction_check,
    validation: campaign.validation,
    diagnosis: campaign.diagnosis,
    affine: campaign.affine_session_id,
    sha: campaign.profile_sha256,
  });
  if (signature === campaignSignature) return;
  campaignSignature = signature;
  const rail = campaign.stages.map((stage) => `<span class="${stage.state}">${stage.title}</span>`).join("");
  const gateClass = campaign.gate.passed ? "campaign-gate open" : "campaign-gate";
  body.innerHTML = `
    <div class="campaign-rail">${rail}</div>
    <div class="campaign-copy">
      <div>
        <p><strong>O que fazer.</strong> ${campaign.copy.what}</p>
        <p><strong>Por quê.</strong> ${campaign.copy.why}</p>
      </div>
      <div class="${gateClass}"><strong>Para avançar.</strong> ${campaign.gate.reason}</div>
    </div>
    <div class="campaign-actions" id="campaign-actions"></div>
    <p class="campaign-live" id="campaign-live">${liveText}</p>
    <p class="focus-meter" id="focus-meter"></p>
  `;
  paintFocus(campaign);
  const actions = document.getElementById("campaign-actions");
  const stage = campaign.stage;
  if (campaign.focus?.drift) {
    addButton(actions, "Recomeçar campanha", () => campaignPost("/api/campaigns/active/restart"));
    return;
  }
  if (stage === "fix_hardware") {
    const labels = [
      ["camera_fixed", "Câmera fixa no suporte"],
      ["lens_focus_locked", "Lente e foco travados como na produção"],
      ["production_resolution", "Resolução e downscale iguais aos da produção"],
      ["single_stapi_client", "Nenhum outro programa está com a câmera"],
    ];
    labels.forEach(([key, label]) => {
      const box = document.createElement("label");
      const disabled = key === "lens_focus_locked" && !campaign.focus?.stable && !campaign.checks[key];
      box.innerHTML = `<input type="checkbox" data-check="${key}" ${campaign.checks[key] ? "checked" : ""} ${disabled ? "disabled" : ""}> ${label}`;
      actions.appendChild(box);
    });
    actions.querySelectorAll("input").forEach((input) => {
      input.addEventListener("change", () => {
        const checks = { ...campaign.checks };
        actions.querySelectorAll("input").forEach((node) => {
          checks[node.dataset.check] = node.checked;
        });
        campaignPost("/api/campaigns/active/checks", checks);
      });
    });
  }
  if (stage === "capture_intrinsic") {
    actions.innerHTML = `<span>${campaign.accepted_count} aceitas · ${campaign.coverage.suggestion}</span>`;
    addButton(actions, "Aceitar imagem", () => campaignPost("/api/campaigns/active/captures", { decision: "accepted" }));
    addButton(actions, "Rejeitar", () => campaignPost("/api/campaigns/active/captures", { decision: "rejected", reason: "Imagem ruim" }), true);
  }
  if (stage === "review_reprojection") {
    addButton(actions, "Calcular K e distorção", () => campaignPost("/api/campaigns/active/intrinsic/solve"));
    addButton(actions, "Reprojeção inspecionada", () => campaignPost("/api/campaigns/active/reprojection/confirm"), true);
    const list = document.createElement("ul");
    list.className = "campaign-list";
    (campaign.reprojection.images || []).forEach((image) => {
      const item = document.createElement("li");
      const error = image.error_px === null || image.error_px === undefined ? "—" : Number(image.error_px).toFixed(3);
      item.textContent = `${image.id.slice(0, 8)} · ${error} px${image.suspect ? " · suspeita" : ""}`;
      if (image.suspect) {
        const exclude = document.createElement("button");
        exclude.className = "button button-secondary";
        exclude.textContent = "Excluir";
        exclude.addEventListener("click", () => campaignPost(`/api/campaigns/active/captures/${image.id}/exclude`));
        item.appendChild(exclude);
      }
      list.appendChild(item);
    });
    actions.appendChild(list);
  }
  if (stage === "confirm_undistort") {
    addButton(actions, "A cadeia seguinte usa a imagem corrigida", () => campaignPost("/api/campaigns/active/undistort/confirm"));
  }
  if (stage === "teach_frame") {
    addButton(actions, "Gravar origem", () => campaignPost("/api/campaigns/active/frame/origin"));
    addButton(actions, "Gravar +X", () => campaignPost("/api/campaigns/active/frame/plus_x"), true);
    addButton(actions, "Gravar +Y", () => campaignPost("/api/campaigns/active/frame/plus_y"), true);
    const note = document.createElement("span");
    note.textContent = app.config?.mode?.plc === "synthetic"
      ? "No simulador, a pose CIP já está no ponto certo a cada gravação."
      : "Encoste o TCP no ponto e grave. O app só lê a pose, não move o robô.";
    actions.appendChild(note);
  }
  if (stage === "solve_extrinsic") {
    addButton(actions, "Aceitar imagem desta pose", () => campaignPost("/api/campaigns/active/captures", { decision: "accepted" }));
    addButton(actions, "Calcular solvePnP", () => campaignPost("/api/campaigns/active/extrinsic/solve"), true);
  }
  if (stage === "confirm_direction") {
    addButton(actions, "Gravar TCP neste canto", () => campaignPost("/api/campaigns/active/direction/touch"));
    addButton(actions, "Confirmar tabuleiro → câmera", () => campaignPost("/api/campaigns/active/direction/confirm", { sense: "object_to_camera" }), true);
    const touchNote = document.createElement("span");
    touchNote.textContent = app.config?.mode?.plc === "synthetic"
      ? "No simulador, a pose já está no canto a um quadrado da origem."
      : "Encoste o TCP no canto interno a um quadrado da origem, no eixo dos 9 cantos.";
    actions.appendChild(touchNote);
    if (campaign.direction_check) {
      const note = document.createElement("span");
      note.textContent = `Erro do sentido certo ${Number(campaign.direction_check.error_correct_mm).toFixed(1)} mm · invertido ${Number(campaign.direction_check.error_inverted_mm).toFixed(1)} mm`;
      actions.appendChild(note);
    }
  }
  if (stage === "validate_planes") {
    [0, 200, 400].forEach((plane) => {
      const metric = campaign.validation[String(plane)];
      const label = metric ? `Z ${plane} · RMSE ${Number(metric.rmse).toFixed(1)} mm` : `Selecionar Z ${plane}`;
      addButton(actions, label, () => campaignPost("/api/campaigns/active/validation/plane", { plan_z: plane }), plane !== 0);
    });
    addButton(actions, "Aceitar imagem nova deste plano", () => campaignPost("/api/campaigns/active/captures", { decision: "accepted" }));
  }
  if (stage === "affine_mold") {
    addButton(actions, "Abrir coleta do molde", () => campaignPost("/api/campaigns/active/affine/open"));
    if (campaign.affine_session_id) {
      const note = document.createElement("span");
      note.textContent = "Sessão do molde aberta. Use CAPTURAR PONTO na coleta abaixo. O centroide entra corrigido.";
      actions.appendChild(note);
    }
    if (campaign.profile_sha256) {
      const hash = document.createElement("span");
      hash.textContent = `Perfil ${campaign.profile_sha256.slice(0, 12)}`;
      actions.appendChild(hash);
    }
  }
  if (stage === "analyze_residuals" && campaign.diagnosis) {
    const list = document.createElement("ul");
    list.className = "campaign-list";
    campaign.diagnosis.findings.forEach((finding) => {
      const item = document.createElement("li");
      item.textContent = finding;
      list.appendChild(item);
    });
    actions.appendChild(list);
    if (campaign.diagnosis.ransac_available) {
      addButton(actions, "Aplicar RANSAC nos outliers", () => campaignPost("/api/campaigns/active/ransac"));
    }
  }
  if (campaign.gate.passed && stage !== "analyze_residuals") {
    addButton(actions, "Avançar", () => campaignPost("/api/campaigns/active/advance"));
  }
}

function paintFocus(campaign) {
  const node = document.getElementById("focus-meter");
  if (!node || !campaign?.focus) return;
  const focus = campaign.focus;
  if (focus.current === null || focus.current === undefined) {
    node.textContent = focus.status;
    return;
  }
  node.textContent = `Nitidez ${Number(focus.current).toFixed(0)} · máximo ${Number(focus.peak).toFixed(0)} · ${focus.status}`;
}

function addButton(parent, label, onClick, secondary = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = secondary ? "button button-secondary" : "button button-primary";
  button.textContent = label;
  button.addEventListener("click", onClick);
  parent.appendChild(button);
}

document.getElementById("new-campaign").addEventListener("click", () => {
  document.getElementById("campaign-dialog").showModal();
});

document.getElementById("campaign-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = document.getElementById("campaign-name").value.trim();
  const square = Number(document.getElementById("campaign-square").value);
  try {
    await api("/api/campaigns", { method: "POST", body: JSON.stringify({ name, square_size_mm: square }) });
    document.getElementById("campaign-dialog").close();
    campaignSignature = "";
    await refreshState();
    toast("Campanha aberta. Siga a etapa 1.");
  } catch (error) {
    toast(error.message, "error");
  }
});

window.renderCampaign = renderCampaign;

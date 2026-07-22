const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = {
  socket: null,
  condition: "",
  model: "",
  models: [],
  vignette: null,
  rounds: { 1: freshRound(), 2: freshRound(), 3: freshRound() },
  selectedRound: 1,
  activeRound: 1,
  running: false,
  startedAt: null,
  timer: null,
  pendingRestart: false,
  awaitingSimulationStart: false,
  selectedEvaluationTab: "accuracy",
  eventQueue: Promise.resolve(),
};

function freshRound() {
  return {
    messages: [], moderator: null, critique: null, ddx: null, evaluation: null,
    evaluationLoading: false, completed: false, reviewed: false, status: null,
  };
}

const phaseNames = {
  vignette: "Vignette 正在生成病例",
  dialogue: "新一轮问诊开始",
  patient: "Patient 正在回答",
  doctor: "Doctor 正在思考",
  moderator: "Moderator 正在判定",
  ddx: "Doctor 正在整理 DDx",
  critic: "Critic 正在复盘",
  evaluation: "Evaluation 正在运行四组代理评分",
};

function connect() {
  if (state.socket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(state.socket.readyState)) return;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  state.socket = new WebSocket(`${protocol}://${location.host}/ws/simulation`);
  state.socket.onmessage = (message) => {
    const event = JSON.parse(message.data);
    state.eventQueue = state.eventQueue.then(() => handleEvent(event));
  };
  state.socket.onclose = () => {
    if (state.running) {
      setRunning(false, "连接已断开");
      showToast("与服务端的连接已断开，请重新开始。", 5000);
    }
  };
}

function send(action, extra = {}) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
    connect();
    const retry = () => send(action, extra);
    state.socket.addEventListener("open", retry, { once: true });
    return;
  }
  state.socket.send(JSON.stringify({ action, ...extra }));
}

function resetSimulation() {
  state.vignette = null;
  state.rounds = { 1: freshRound(), 2: freshRound(), 3: freshRound() };
  state.selectedRound = 1;
  state.activeRound = 1;
  state.selectedEvaluationTab = "accuracy";
  $("#emptyState").hidden = true;
  $("#simulationView").hidden = false;
  $("#workspace").classList.remove("is-empty");
  $("#vignetteSkeleton").hidden = false;
  $("#vignetteContent").hidden = true;
  $("#vignetteContent").replaceChildren();
  $$(".round-tab").forEach((tab, index) => {
    tab.disabled = index !== 0;
    tab.classList.toggle("active", index === 0);
  });
  renderRound(1);
  renderDDx(1);
  renderEvaluation(1);
}

function startSimulation() {
  const condition = $("#condition").value.trim();
  if (!condition) {
    showToast("请先输入一个 medical condition。", 3000);
    $("#condition").focus();
    return;
  }
  state.condition = condition;
  state.model = $("#modelSelect").value || state.model;
  state.awaitingSimulationStart = true;
  resetSimulation();
  state.startedAt = Date.now();
  startTimer();
  setRunning(true, "准备生成病例");
  send("start", { condition, model: state.model });
  $("#workspace").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function handleEvent(event) {
  switch (event.type) {
    case "simulation_started":
      state.awaitingSimulationStart = false;
      state.model = event.model;
      $("#modelLabel").textContent = event.model;
      $("#footerModel").textContent = event.model;
      break;
    case "phase_started":
      if (event.phase === "evaluation" && event.round) {
        if (state.awaitingSimulationStart) break;
        state.rounds[event.round].evaluationLoading = true;
        if (!state.running) {
          $("#phaseLabel").textContent = "Evaluation 正在后台运行";
          setActiveFlow(event.phase);
        }
        if (event.round === state.selectedRound) renderEvaluation(event.round);
        break;
      }
      state.activeRound = event.round || state.activeRound;
      setRunning(true, phaseNames[event.phase] || event.phase);
      setActiveFlow(event.phase);
      if (event.round) $("#roundLabel").textContent = `${event.round} / 3`;
      if (event.phase === "dialogue" && event.round > 1) {
        enableRound(event.round);
        selectRound(event.round);
      }
      break;
    case "vignette_completed":
      state.vignette = event.vignette;
      renderVignette(event.vignette);
      break;
    case "message_started":
      if (event.round === state.selectedRound) $("#typingIndicator").hidden = false;
      break;
    case "message_completed":
      $("#typingIndicator").hidden = true;
      state.rounds[event.round].messages.push({
        role: event.role,
        content: event.content,
        contentHtml: typeof event.content_html === "string" ? event.content_html : null,
      });
      if (event.round === state.selectedRound) {
        await appendMessage(event.role, event.content, event.index, true, event.content_html);
      }
      updateUtteranceCount();
      break;
    case "moderator_result":
      state.rounds[event.round].moderator = event;
      if (event.round === state.selectedRound) renderModerator(event);
      break;
    case "critique_completed":
      state.rounds[event.round].critique = event.critique;
      if (event.round === state.selectedRound) renderCritic(event.critique, event.round);
      break;
    case "evaluation_completed":
      if (state.awaitingSimulationStart) break;
      state.rounds[event.round].evaluation = event;
      state.rounds[event.round].evaluationLoading = false;
      if (event.round === state.selectedRound) renderEvaluation(event.round);
      break;
    case "ddx_completed":
      state.rounds[event.round].ddx = event.differential_diagnoses;
      if (event.round === state.selectedRound) renderDDx(event.round);
      break;
    case "dialogue_completed":
      Object.assign(state.rounds[event.round], { completed: true, status: event.status, completion: event });
      setRunning(true, "正在生成 DDx、Critic 与 Evaluation");
      if (event.round === state.selectedRound) {
        renderRoundActions(event);
        showDDxLoading();
      }
      if (event.status === "truncated") showToast(event.reason, 5000);
      break;
    case "round_review_ready": {
      if (state.awaitingSimulationStart) break;
      const round = state.rounds[event.round];
      round.reviewed = true;
      round.completion = {
        ...round.completion,
        ...event,
        review_pending: false,
        evaluation_pending: true,
      };
      setRunning(false, event.status === "truncated" ? "已完成截断轮复盘" : `Round ${event.round} Critic 复盘完成`);
      if (event.round === state.selectedRound) {
        renderRoundActions(round.completion);
        renderEvaluation(event.round);
        $("#evaluationPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
      break;
    }
    case "round_review_completed": {
      if (state.awaitingSimulationStart) break;
      const round = state.rounds[event.round];
      round.completion = {
        ...round.completion,
        ...event,
        review_pending: false,
        evaluation_pending: false,
      };
      if (!state.running && event.round === state.activeRound) {
        $("#phaseLabel").textContent = `Round ${event.round} 复盘与评分完成`;
      }
      if (event.round === state.selectedRound) {
        renderRoundActions(round.completion);
        renderEvaluation(event.round);
      }
      break;
    }
    case "error":
      state.awaitingSimulationStart = false;
      setRunning(false, "发生错误");
      showToast(event.message || "模拟失败", 6000);
      break;
    case "stopped":
      state.awaitingSimulationStart = false;
      setRunning(false, "已停止");
      if (state.pendingRestart) {
        state.pendingRestart = false;
        startSimulation();
      } else {
        showToast(event.reason || "已停止模拟", 3000);
      }
      break;
  }
}

function renderVignette(vignette) {
  const labels = {
    condition: "Condition", summary: "Summary", demographics: "Demographics",
    symptoms: "Symptoms", past_medical_history: "Past Medical History",
    past_surgical_history: "Past Surgical History", past_social_history: "Past Social History",
    medication: "Medication", allergy: "Allergy", family_history: "Family History",
    patient_questions: "Patient Questions", ground_truth_diagnosis: "Ground-truth Diagnosis",
    accepted_differential_diagnoses: "Accepted Differential",
    reference_management_plan: "Reference Management Plan",
  };
  const sensitive = new Set([
    "ground_truth_diagnosis", "accepted_differential_diagnoses", "reference_management_plan",
  ]);
  const container = $("#vignetteContent");
  container.replaceChildren();
  Object.entries(labels).forEach(([key, label]) => {
    const section = document.createElement("section");
    section.className = `vignette-field${sensitive.has(key) ? " sensitive" : ""}`;
    const heading = document.createElement("small");
    heading.textContent = label;
    section.appendChild(heading);
    const value = vignette[key];
    if (Array.isArray(value)) {
      const list = document.createElement("ul");
      value.forEach((item) => { const li = document.createElement("li"); li.textContent = item; list.appendChild(li); });
      section.appendChild(list);
    } else {
      const paragraph = document.createElement("p");
      paragraph.textContent = value || "N/A";
      section.appendChild(paragraph);
    }
    container.appendChild(section);
  });
  $("#vignetteSkeleton").hidden = true;
  container.hidden = false;
}

async function appendMessage(role, content, index, animate = true, contentHtml = null) {
  const timeline = $("#dialogueTimeline");
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "doctor" ? "D" : "P";
  const body = document.createElement("div");
  body.className = "message-body";
  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = `${role === "doctor" ? "Doctor" : "Patient"} · #${index}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  body.append(meta, bubble);
  article.append(avatar, body);
  timeline.appendChild(article);
  const safeHtml = role === "doctor" && typeof contentHtml === "string" ? contentHtml : null;
  const animatedText = safeHtml ? readableTextFromHtml(safeHtml) : content;
  if (animate && animatedText.length > 1) await typeText(bubble, animatedText);
  else bubble.textContent = animatedText;
  if (safeHtml) {
    bubble.classList.add("rich-content");
    bubble.innerHTML = safeHtml;
  }
  article.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function readableTextFromHtml(contentHtml) {
  const template = document.createElement("template");
  template.innerHTML = contentHtml;
  template.content.querySelectorAll("p, blockquote, li").forEach((element) => {
    element.appendChild(document.createTextNode("\n"));
  });
  return template.content.textContent.replace(/\n{3,}/g, "\n\n").trim();
}

function typeText(element, content) {
  const duration = Math.min(1500, Math.max(180, content.length * 14));
  return new Promise((resolve) => {
    const start = performance.now();
    const tick = (now) => {
      const count = Math.min(content.length, Math.ceil(((now - start) / duration) * content.length));
      element.textContent = content.slice(0, count);
      if (count < content.length) requestAnimationFrame(tick);
      else resolve();
    };
    requestAnimationFrame(tick);
  });
}

function renderModerator(result) {
  const card = $("#moderatorCard");
  const value = (flag) => flag ? "已满足" : "未满足";
  card.innerHTML = "";
  const title = document.createElement("div");
  title.className = "card-title";
  title.innerHTML = `<span>M</span> Moderator · ${result.ended ? "结束本轮" : "继续问诊"}`;
  const reason = document.createElement("div");
  reason.textContent = result.reason;
  const dl = document.createElement("dl");
  [
    ["诊断判断", result.diagnosis_complete],
    ["治疗计划", result.treatment_plan_complete],
    ["患者问题", result.patient_questions_resolved],
  ].forEach(([label, flag]) => {
    const box = document.createElement("div");
    const dt = document.createElement("dt"); dt.textContent = label;
    const dd = document.createElement("dd"); dd.textContent = value(flag);
    box.append(dt, dd); dl.appendChild(box);
  });
  card.append(title, reason, dl);
  card.hidden = false;
}

function renderCritic(content, round) {
  const card = $("#criticCard");
  card.innerHTML = "";
  const title = document.createElement("div");
  title.className = "card-title";
  title.innerHTML = `<span>C</span> Critic · Round ${round} 复盘`;
  const text = document.createElement("div");
  text.textContent = content;
  card.append(title, text);
  card.hidden = false;
}

function showDDxLoading() {
  $("#ddxEmpty").hidden = true;
  $("#ddxList").hidden = true;
  $("#ddxSkeleton").hidden = false;
}

function renderDDx(round) {
  const diagnoses = state.rounds[round].ddx;
  const empty = $("#ddxEmpty");
  const skeleton = $("#ddxSkeleton");
  const list = $("#ddxList");
  skeleton.hidden = true;
  list.replaceChildren();
  if (!diagnoses) {
    empty.hidden = false;
    list.hidden = true;
    return;
  }
  diagnoses.forEach((diagnosis, index) => {
    const item = document.createElement("li");
    item.textContent = diagnosis;
    if (index === 0) {
      const label = document.createElement("small");
      label.textContent = "Most likely";
      item.appendChild(label);
    }
    list.appendChild(item);
  });
  empty.hidden = true;
  list.hidden = false;
}

const evaluationGroupNames = {
  accuracy: "Accuracy",
  patient_actor: "Patient Actor",
  specialist: "Specialist",
  auto_paces: "Auto PACES",
};

const matchLevelNames = {
  exact: "Exact",
  synonym: "Synonym / abbreviation",
  more_specific: "More specific",
  highly_related: "Highly related",
  no_match: "No match",
};

function renderEvaluation(round) {
  const data = state.rounds[round];
  const evaluation = data.evaluation;
  const empty = $("#evaluationEmpty");
  const skeleton = $("#evaluationSkeleton");
  const content = $("#evaluationContent");

  if (!evaluation) {
    empty.hidden = data.evaluationLoading;
    skeleton.hidden = !data.evaluationLoading;
    content.hidden = true;
    renderRoundActions(data.completion || {});
    return;
  }

  empty.hidden = true;
  skeleton.hidden = true;
  content.hidden = false;

  const statusNames = { complete: "Complete", partial: "Partial", failed: "Failed" };
  const status = $("#evaluationStatus");
  status.textContent = statusNames[evaluation.status] || evaluation.status;
  status.className = `evaluation-status ${evaluation.status}`;
  const completedCount = Object.keys(evaluationGroupNames)
    .filter((name) => evaluation[name]).length;
  $("#evaluationCoverage").textContent = `${completedCount} / 4 groups`;

  renderEvaluationErrors(evaluation.errors || {});
  renderTopKMatrix(evaluation.accuracy);
  renderDistributions(evaluation);
  renderAccuracyEvaluation(evaluation.accuracy);
  renderQualityEvaluation($("#patientActorEvaluation"), evaluation.patient_actor, "Patient-actor proxy · 26 axes");
  renderQualityEvaluation($("#specialistEvaluation"), evaluation.specialist, "Specialist physician proxy · 32 axes");
  renderQualityEvaluation($("#autoPacesEvaluation"), evaluation.auto_paces, "Simulated dialogue PACES · 4 axes");
  selectEvaluationTab(state.selectedEvaluationTab);
  renderRoundActions(data.completion || {});
}

function renderEvaluationErrors(errors) {
  const container = $("#evaluationErrors");
  container.replaceChildren();
  const entries = Object.entries(errors);
  if (!entries.length) {
    container.hidden = true;
    return;
  }
  const heading = document.createElement("b");
  heading.textContent = "部分评审未完成";
  container.appendChild(heading);
  entries.forEach(([name, error]) => {
    const item = document.createElement("div");
    const label = document.createElement("strong");
    label.textContent = evaluationGroupNames[name] || name;
    const detail = document.createElement("span");
    detail.textContent = `${error.category || "error"} · ${error.message || error.code || "Unknown error"}`;
    item.append(label, detail);
    container.appendChild(item);
  });
  container.hidden = false;
}

function renderTopKMatrix(accuracy) {
  const matrix = $("#topKMatrix");
  matrix.replaceChildren();
  ["Comparison", "Top-1", "Top-3", "Top-10"].forEach((label) => {
    const cell = document.createElement("div");
    cell.className = "topk-header";
    cell.textContent = label;
    matrix.appendChild(cell);
  });
  const rows = [
    ["Ground truth", accuracy?.ground_truth_hits],
    ["Accepted differential", accuracy?.accepted_differential_hits],
  ];
  rows.forEach(([label, hits]) => {
    const name = document.createElement("div");
    name.className = "topk-label";
    name.textContent = label;
    matrix.appendChild(name);
    ["top_1", "top_3", "top_10"].forEach((key) => {
      const cell = document.createElement("div");
      if (!hits) {
        cell.className = "topk-result unavailable";
        cell.textContent = "N/A";
      } else {
        cell.className = `topk-result ${hits[key] ? "hit" : "miss"}`;
        cell.textContent = hits[key] ? "命中" : "未命中";
      }
      matrix.appendChild(cell);
    });
  });
}

function polarityCounts(result) {
  const counts = { positive: 0, neutral: 0, negative: 0, na: 0 };
  (result?.criteria || []).forEach((criterion) => {
    if (criterion.polarity in counts) counts[criterion.polarity] += 1;
  });
  return counts;
}

function renderDistributions(evaluation) {
  const grid = $("#distributionGrid");
  grid.replaceChildren();
  [
    ["Patient Actor", evaluation.patient_actor],
    ["Specialist", evaluation.specialist],
    ["Auto PACES", evaluation.auto_paces],
  ].forEach(([label, result]) => {
    const card = document.createElement("section");
    card.className = "distribution-card";
    const title = document.createElement("b");
    title.textContent = label;
    card.appendChild(title);
    if (!result) {
      const unavailable = document.createElement("span");
      unavailable.className = "distribution-unavailable";
      unavailable.textContent = "Unavailable";
      card.appendChild(unavailable);
    } else {
      const counts = polarityCounts(result);
      const row = document.createElement("div");
      row.className = "distribution-counts";
      [
        ["positive", "正"], ["neutral", "中"], ["negative", "负"], ["na", "N/A"],
      ].forEach(([key, name]) => {
        const item = document.createElement("span");
        item.className = key;
        const value = document.createElement("strong");
        value.textContent = counts[key];
        const caption = document.createElement("small");
        caption.textContent = name;
        item.append(value, caption);
        row.appendChild(item);
      });
      card.appendChild(row);
    }
    grid.appendChild(card);
  });
}

function renderAccuracyEvaluation(accuracy) {
  const container = $("#accuracyEvaluation");
  container.replaceChildren();
  if (!accuracy) {
    renderUnavailableEvaluation(container, "Accuracy 评审未返回有效结果。");
    return;
  }
  const intro = document.createElement("div");
  intro.className = "tab-panel-intro";
  const title = document.createElement("h3");
  title.textContent = "DDx semantic matching";
  const note = document.createElement("p");
  note.textContent = "每个候选分别与 ground truth 和 accepted differential 比较；Top-10 使用全部已有 DDx。";
  intro.append(title, note);
  container.appendChild(intro);

  const list = document.createElement("div");
  list.className = "accuracy-candidate-list";
  accuracy.candidates.forEach((candidate) => {
    const details = document.createElement("details");
    details.className = "accuracy-candidate";
    const summary = document.createElement("summary");
    const rank = document.createElement("span");
    rank.className = "candidate-rank";
    rank.textContent = candidate.rank;
    const diagnosis = document.createElement("b");
    diagnosis.textContent = candidate.candidate;
    const badges = document.createElement("span");
    badges.className = "candidate-match-badges";
    badges.append(
      createMatchBadge("GT", candidate.ground_truth.level),
      createMatchBadge("Accepted", candidate.accepted_differential.level),
    );
    summary.append(rank, diagnosis, badges);

    const body = document.createElement("div");
    body.className = "candidate-match-details";
    body.append(
      createMatchDetail("Ground truth", candidate.ground_truth),
      createMatchDetail("Accepted differential", candidate.accepted_differential),
    );
    details.append(summary, body);
    list.appendChild(details);
  });
  container.appendChild(list);
}

function createMatchBadge(prefix, level) {
  const badge = document.createElement("span");
  badge.className = `match-badge ${level === "no_match" ? "no-match" : "matched"}`;
  badge.textContent = `${prefix}: ${matchLevelNames[level] || level}`;
  return badge;
}

function createMatchDetail(label, match) {
  const section = document.createElement("section");
  const title = document.createElement("b");
  title.textContent = label;
  const target = document.createElement("small");
  target.textContent = match.matched_diagnosis
    ? `${matchLevelNames[match.level] || match.level} · ${match.matched_diagnosis}`
    : matchLevelNames[match.level] || match.level;
  const rationale = document.createElement("p");
  rationale.textContent = match.rationale;
  section.append(title, target, rationale);
  return section;
}

function renderQualityEvaluation(container, result, titleText) {
  container.replaceChildren();
  if (!result) {
    renderUnavailableEvaluation(container, `${titleText} 未返回有效结果。`);
    return;
  }
  const intro = document.createElement("div");
  intro.className = "tab-panel-intro";
  const title = document.createElement("h3");
  title.textContent = titleText;
  const note = document.createElement("p");
  note.textContent = "展开任一评分轴查看模型代理给出的 transcript 证据。";
  intro.append(title, note);
  container.appendChild(intro);

  const groups = new Map();
  result.criteria.forEach((criterion) => {
    if (!groups.has(criterion.group)) groups.set(criterion.group, []);
    groups.get(criterion.group).push(criterion);
  });
  groups.forEach((criteria, groupName) => {
    const group = document.createElement("section");
    group.className = "rubric-group";
    const heading = document.createElement("div");
    heading.className = "rubric-group-heading";
    const name = document.createElement("h4");
    name.textContent = groupName;
    const count = document.createElement("span");
    count.textContent = `${criteria.length} axes`;
    heading.append(name, count);
    group.appendChild(heading);
    criteria.forEach((criterion) => group.appendChild(createCriterionDetails(criterion)));
    container.appendChild(group);
  });
}

function createCriterionDetails(criterion) {
  const details = document.createElement("details");
  details.className = `criterion-row ${criterion.polarity}`;
  const summary = document.createElement("summary");
  const label = document.createElement("div");
  label.className = "criterion-label";
  const name = document.createElement("b");
  name.textContent = criterion.label;
  const id = document.createElement("small");
  id.textContent = criterion.criterion_id;
  label.append(name, id);

  const response = document.createElement("div");
  response.className = "criterion-response";
  const responseText = document.createElement("span");
  responseText.textContent = criterion.response_label;
  const raw = document.createElement("strong");
  raw.textContent = criterion.raw_score === null
    ? "N/A"
    : `${criterion.raw_score} / ${criterion.scale_max}`;
  const track = document.createElement("i");
  const fill = document.createElement("i");
  if (criterion.raw_score !== null) {
    fill.style.width = `${Math.max(0, Math.min(100, (criterion.raw_score / criterion.scale_max) * 100))}%`;
  }
  track.appendChild(fill);
  response.append(responseText, raw, track);
  summary.append(label, response);

  const evidence = document.createElement("div");
  evidence.className = "criterion-evidence";
  const caption = document.createElement("b");
  caption.textContent = "Evidence";
  const text = document.createElement("p");
  text.textContent = criterion.evidence;
  evidence.append(caption, text);
  details.append(summary, evidence);
  return details;
}

function renderUnavailableEvaluation(container, message) {
  const empty = document.createElement("div");
  empty.className = "evaluation-unavailable";
  empty.textContent = message;
  container.appendChild(empty);
}

function selectEvaluationTab(name) {
  if (!evaluationGroupNames[name]) name = "accuracy";
  state.selectedEvaluationTab = name;
  $$(".evaluation-tab").forEach((tab) => {
    const active = tab.dataset.evaluationTab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  $$(".evaluation-tab-panel").forEach((panel) => {
    panel.hidden = panel.dataset.evaluationPanel !== name;
  });
}

function renderRound(round) {
  const data = state.rounds[round];
  $("#dialogueTitle").textContent = `Round ${round} 对话`;
  $("#dialogueTimeline").replaceChildren();
  $("#moderatorCard").hidden = true;
  $("#criticCard").hidden = true;
  $("#roundActions").hidden = true;
  data.messages.forEach((message, index) => {
    appendMessage(message.role, message.content, index + 1, false, message.contentHtml);
  });
  if (data.moderator) renderModerator(data.moderator);
  if (data.critique) renderCritic(data.critique, round);
  if (data.completion) renderRoundActions(data.completion);
  renderDDx(round);
  renderEvaluation(round);
  updateUtteranceCount();
}

function renderRoundActions(completion) {
  const actions = $("#roundActions");
  const isLatestRound = Number(completion.round) === state.activeRound;
  const canRefine = completion.can_refine && completion.status === "completed" && isLatestRound;
  actions.hidden = !canRefine;
  $("#refineButton").disabled = state.running || !canRefine;
}

function selectRound(round) {
  const tab = $(`.round-tab[data-round="${round}"]`);
  if (!tab || tab.disabled) return;
  state.selectedRound = round;
  $$(".round-tab").forEach((item) => item.classList.toggle("active", Number(item.dataset.round) === round));
  renderRound(round);
}

function enableRound(round) {
  if (!round) return;
  const tab = $(`.round-tab[data-round="${round}"]`);
  if (tab) tab.disabled = false;
}

function updateUtteranceCount() {
  $("#utteranceCount").textContent = `${state.rounds[state.selectedRound].messages.length} / 30`;
}

function setActiveFlow(phase) {
  $$(".flow-node").forEach((node) => node.classList.toggle("active", node.dataset.flow === phase));
}

function setRunning(running, label) {
  state.running = running;
  $("#phaseLabel").textContent = label;
  $("#liveDot").classList.toggle("running", running);
  $("#startButton").disabled = running || state.models.length === 0;
  $("#modelSelect").disabled = running || state.models.length === 0;
  renderRoundActions(state.rounds[state.selectedRound].completion || {});
  if (!running) $("#typingIndicator").hidden = true;
}

function startTimer() {
  clearInterval(state.timer);
  const update = () => {
    const seconds = Math.floor((Date.now() - state.startedAt) / 1000);
    const minutes = String(Math.floor(seconds / 60)).padStart(2, "0");
    $("#elapsedLabel").textContent = `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
  };
  update();
  state.timer = setInterval(update, 1000);
}

let toastTimer;
function showToast(message, duration = 4000) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, duration);
}

let autocompleteTimer;
$("#condition").addEventListener("input", () => {
  clearTimeout(autocompleteTimer);
  const query = $("#condition").value.trim();
  if (query.length < 2) { $("#suggestions").hidden = true; return; }
  autocompleteTimer = setTimeout(async () => {
    try {
      const response = await fetch(`/api/diseases?q=${encodeURIComponent(query)}&limit=10`);
      const { items } = await response.json();
      const box = $("#suggestions");
      box.replaceChildren();
      items.forEach((name) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = name;
        button.addEventListener("click", () => { $("#condition").value = name; box.hidden = true; });
        box.appendChild(button);
      });
      box.hidden = items.length === 0;
    } catch { $("#suggestions").hidden = true; }
  }, 180);
});

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw new Error("model catalog unavailable");
    const data = await response.json();
    if (!Array.isArray(data.items) || data.items.length === 0) {
      throw new Error("empty model catalog");
    }
    state.models = data.items;
    state.model = data.default_model;
    const select = $("#modelSelect");
    select.replaceChildren();
    const ordered = [...data.items].sort((a, b) => {
      if (a.name === data.default_model) return -1;
      if (b.name === data.default_model) return 1;
      return 0;
    });
    ordered.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = model.real_name && model.real_name !== model.name
        ? `${model.name} · ${model.real_name}`
        : model.name;
      option.selected = model.name === data.default_model;
      select.appendChild(option);
    });
    select.disabled = false;
    $("#startButton").disabled = false;
    updateModelMeta();
  } catch {
    state.models = [];
    state.model = "";
    const select = $("#modelSelect");
    select.innerHTML = '<option value="">模型配置不可用</option>';
    select.disabled = true;
    $("#startButton").disabled = true;
    $("#modelMeta").textContent = "请检查 config/model_apis.json";
    showToast("模型目录载入失败，请检查服务端模型 API 配置。", 5000);
  }
}

function updateModelMeta() {
  const selected = state.models.find((model) => model.name === $("#modelSelect").value);
  if (!selected) return;
  const context = selected.context_length >= 1000000
    ? `${(selected.context_length / 1048576).toFixed(0)}M context`
    : selected.context_length > 0
      ? `${Math.round(selected.context_length / 1000)}K context`
      : "context unknown";
  const capabilities = [selected.image_input ? "vision" : "text", selected.support_stream ? "stream" : "non-stream"];
  $("#modelMeta").textContent = `${selected.real_name} · ${context} · ${capabilities.join(" · ")}`;
}

document.addEventListener("click", (event) => {
  if (!event.target.closest(".autocomplete")) $("#suggestions").hidden = true;
});
$("#condition").addEventListener("keydown", (event) => { if (event.key === "Enter") startSimulation(); });
$("#startButton").addEventListener("click", startSimulation);
$("#modelSelect").addEventListener("change", updateModelMeta);
$("#restartButton").addEventListener("click", () => {
  if (state.running) {
    state.pendingRestart = true;
    send("stop");
  } else {
    startSimulation();
  }
});
$("#stopButton").addEventListener("click", () => send("stop"));
$("#refineButton").addEventListener("click", () => { setRunning(true, "正在生成改进轮"); send("refine"); });
$$('[data-condition]').forEach((button) => button.addEventListener("click", () => { $("#condition").value = button.dataset.condition; }));
$$(".round-tab").forEach((tab) => tab.addEventListener("click", () => selectRound(Number(tab.dataset.round))));
$$(".evaluation-tab").forEach((tab) => tab.addEventListener("click", () => selectEvaluationTab(tab.dataset.evaluationTab)));

loadModels();
connect();

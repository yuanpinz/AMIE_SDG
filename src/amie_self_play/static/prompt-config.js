const $ = (selector) => document.querySelector(selector);

const state = {
  agents: [],
  selectedAgent: null,
  source: "",
  busy: false,
  toastTimer: null,
};

const tokenStorageKey = "amiePromptAdminToken";
const fieldLabels = {
  system: "System Prompt",
  user: "User Prompt",
  improvement: "改进轮补充 Prompt",
};

function adminHeaders(withJson = false) {
  const headers = {};
  const token = sessionStorage.getItem(tokenStorageKey) || "";
  if (token) headers["X-AMIE-ADMIN-TOKEN"] = token;
  if (withJson) headers["Content-Type"] = "application/json";
  return headers;
}

async function apiRequest(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...adminHeaders(Boolean(options.body)), ...(options.headers || {}) },
  });
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    if (response.status === 401) $("#authPanel").hidden = false;
    const detail = data?.detail;
    const message = typeof detail === "string"
      ? detail
      : detail
        ? JSON.stringify(detail)
        : `请求失败（HTTP ${response.status}）`;
    throw new Error(message);
  }
  return data;
}

function currentAgent() {
  return state.agents.find((agent) => agent.name === state.selectedAgent) || null;
}

function collectEditorValues() {
  return Object.fromEntries(
    [...document.querySelectorAll("textarea[data-prompt-field]")].map((textarea) => [
      textarea.dataset.promptField,
      textarea.value,
    ]),
  );
}

function hasUnsavedChanges() {
  const agent = currentAgent();
  if (!agent || $("#editorContent").hidden) return false;
  const values = collectEditorValues();
  return agent.fields.some((field) => values[field.name] !== field.value);
}

function setBusy(busy) {
  state.busy = busy;
  syncDirtyState();
  document.querySelectorAll(".agent-nav-button").forEach((button) => {
    button.disabled = busy;
  });
}

function syncDirtyState() {
  const dirty = hasUnsavedChanges();
  const disabled = state.busy || !dirty;
  $("#saveButton").disabled = disabled;
  $("#saveButtonBottom").disabled = disabled;
  $("#resetButton").disabled = state.busy;
  $("#resetButtonBottom").disabled = state.busy;
  $("#saveState").textContent = dirty ? "有尚未保存的修改" : "当前没有未保存修改";
  $("#saveState").classList.toggle("dirty", dirty);
  const activeButton = document.querySelector(".agent-nav-button.active");
  if (activeButton) activeButton.classList.toggle("unsaved", dirty);
}

function showToast(message, duration = 3600) {
  const toast = $("#promptToast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => { toast.hidden = true; }, duration);
}

function renderAgentList() {
  const list = $("#agentList");
  list.replaceChildren();
  state.agents.forEach((agent, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "agent-nav-button";
    button.dataset.agent = agent.name;
    button.classList.toggle("active", agent.name === state.selectedAgent);

    const mark = document.createElement("span");
    mark.className = "agent-nav-mark";
    mark.textContent = agent.label === "Doctor DDx" ? "Dx" : agent.label[0];
    const copy = document.createElement("span");
    copy.className = "agent-nav-copy";
    const label = document.createElement("b");
    label.textContent = agent.label;
    const description = document.createElement("small");
    description.textContent = agent.description;
    copy.append(label, description);
    const status = document.createElement("i");
    status.className = agent.modified ? "modified-dot" : "default-dot";
    status.title = agent.modified ? "已自定义" : "使用默认值";
    button.append(mark, copy, status);
    button.addEventListener("click", () => selectAgent(agent.name));
    button.style.setProperty("--agent-index", index);
    list.appendChild(button);
  });
  $("#agentCount").textContent = `${state.agents.length} 个`;
}

function fieldLabel(name) {
  return fieldLabels[name] || name;
}

function renderEditor() {
  const agent = currentAgent();
  if (!agent) return;
  $("#editorLoading").hidden = true;
  $("#editorError").hidden = true;
  $("#editorContent").hidden = false;
  $("#agentMark").textContent = agent.label === "Doctor DDx" ? "Dx" : agent.label[0];
  $("#agentTitle").textContent = agent.label;
  $("#agentDescription").textContent = agent.description;
  const status = $("#agentStatus");
  status.textContent = agent.modified ? "已自定义" : "默认配置";
  status.classList.toggle("modified", agent.modified);

  const fields = $("#promptFields");
  fields.replaceChildren();
  agent.fields.forEach((field) => {
    const section = document.createElement("section");
    section.className = "prompt-field";

    const heading = document.createElement("div");
    heading.className = "prompt-field-heading";
    const title = document.createElement("div");
    const kicker = document.createElement("span");
    kicker.textContent = field.name.toUpperCase();
    const label = document.createElement("h3");
    label.textContent = fieldLabel(field.name);
    title.append(kicker, label);
    const count = document.createElement("small");
    count.textContent = `${field.value.length.toLocaleString()} 字符`;
    heading.append(title, count);

    const textarea = document.createElement("textarea");
    textarea.dataset.promptField = field.name;
    textarea.value = field.value;
    textarea.spellcheck = false;
    textarea.setAttribute("aria-label", `${agent.label} ${fieldLabel(field.name)}`);
    const lineCount = field.value.split("\n").length;
    textarea.rows = Math.min(28, Math.max(9, lineCount + 1));
    textarea.addEventListener("input", () => {
      count.textContent = `${textarea.value.length.toLocaleString()} 字符`;
      syncDirtyState();
    });

    const fieldMeta = document.createElement("div");
    fieldMeta.className = "prompt-field-meta";
    const variables = document.createElement("div");
    variables.className = "placeholder-list";
    const variableLabel = document.createElement("span");
    variableLabel.textContent = "必要变量";
    variables.appendChild(variableLabel);
    if (field.required_placeholders.length) {
      field.required_placeholders.forEach((name) => {
        const code = document.createElement("code");
        code.textContent = "${" + name + "}";
        variables.appendChild(code);
      });
    } else {
      const empty = document.createElement("em");
      empty.textContent = "无";
      variables.appendChild(empty);
    }

    const details = document.createElement("details");
    details.className = "default-preview";
    const summary = document.createElement("summary");
    summary.textContent = "查看此字段的内置默认值";
    const pre = document.createElement("pre");
    pre.textContent = field.default;
    details.append(summary, pre);
    fieldMeta.append(variables, details);
    section.append(heading, textarea, fieldMeta);
    fields.appendChild(section);
  });
  syncDirtyState();
}

function selectAgent(name) {
  if (name === state.selectedAgent) return;
  if (hasUnsavedChanges() && !window.confirm("当前 Agent 有未保存修改，确定放弃并切换吗？")) return;
  state.selectedAgent = name;
  renderAgentList();
  renderEditor();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function applyCatalog(data, preferredAgent = null) {
  state.agents = Array.isArray(data.agents) ? data.agents : [];
  state.source = data.source || "";
  const available = new Set(state.agents.map((agent) => agent.name));
  state.selectedAgent = available.has(preferredAgent)
    ? preferredAgent
    : available.has(state.selectedAgent)
      ? state.selectedAgent
      : state.agents[0]?.name || null;
  $("#configSource").textContent = state.source || "—";
  $("#configSource").title = state.source;
  $("#authPanel").hidden = true;
  renderAgentList();
  renderEditor();
}

function showEditorError(message) {
  $("#editorLoading").hidden = true;
  $("#editorContent").hidden = true;
  $("#editorError").hidden = false;
  $("#editorErrorMessage").textContent = message;
}

async function loadCatalog() {
  $("#editorError").hidden = true;
  $("#editorLoading").hidden = false;
  try {
    const data = await apiRequest("/api/admin/prompts");
    applyCatalog(data);
  } catch (error) {
    showEditorError(error.message);
  }
}

async function saveCurrentAgent() {
  const agent = currentAgent();
  if (!agent || state.busy || !hasUnsavedChanges()) return;
  setBusy(true);
  try {
    const data = await apiRequest(`/api/admin/prompts/${encodeURIComponent(agent.name)}`, {
      method: "PUT",
      body: JSON.stringify({ values: collectEditorValues() }),
    });
    applyCatalog(data, agent.name);
    showToast(data.message || `${agent.label} 提示词已保存`);
  } catch (error) {
    showToast(error.message, 6000);
  } finally {
    setBusy(false);
  }
}

async function resetCurrentAgent() {
  const agent = currentAgent();
  if (!agent || state.busy) return;
  const confirmed = window.confirm(
    `确定将 ${agent.label} 的全部提示词恢复为仓库内置默认值吗？\n\n其他 Agent 不会受影响。`,
  );
  if (!confirmed) return;
  setBusy(true);
  try {
    const data = await apiRequest(`/api/admin/prompts/${encodeURIComponent(agent.name)}/reset`, {
      method: "POST",
    });
    applyCatalog(data, agent.name);
    showToast(data.message || `${agent.label} 已恢复默认`);
  } catch (error) {
    showToast(error.message, 6000);
  } finally {
    setBusy(false);
  }
}

function applyAdminToken() {
  const token = $("#adminToken").value.trim();
  if (token) sessionStorage.setItem(tokenStorageKey, token);
  else sessionStorage.removeItem(tokenStorageKey);
  loadCatalog();
}

$("#saveButton").addEventListener("click", saveCurrentAgent);
$("#saveButtonBottom").addEventListener("click", saveCurrentAgent);
$("#resetButton").addEventListener("click", resetCurrentAgent);
$("#resetButtonBottom").addEventListener("click", resetCurrentAgent);
$("#retryButton").addEventListener("click", loadCatalog);
$("#applyToken").addEventListener("click", applyAdminToken);
$("#adminToken").addEventListener("keydown", (event) => {
  if (event.key === "Enter") applyAdminToken();
});

window.addEventListener("beforeunload", (event) => {
  if (!hasUnsavedChanges()) return;
  event.preventDefault();
  event.returnValue = "";
});

$("#adminToken").value = sessionStorage.getItem(tokenStorageKey) || "";
loadCatalog();

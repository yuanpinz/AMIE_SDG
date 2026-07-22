const $ = (selector) => document.querySelector(selector);

let mode = "login";

function destination() {
  const candidate = new URLSearchParams(window.location.search).get("next") || "/";
  return candidate.startsWith("/") && !candidate.startsWith("//") ? candidate : "/";
}

function setMode(nextMode) {
  mode = nextMode;
  const registering = mode === "register";
  $("#authTitle").textContent = registering ? "创建账户" : "登录账户";
  $("#authSubtitle").textContent = registering
    ? "每个账户会获得一份独立的提示词配置。"
    : "继续进入你的模拟工作区和个人提示词配置。";
  $("#confirmGroup").hidden = !registering;
  $("#confirmPassword").required = registering;
  $("#password").autocomplete = registering ? "new-password" : "current-password";
  $("#authSubmit").textContent = registering ? "注册并进入" : "登录";
  $("#authError").hidden = true;
  document.querySelectorAll("[data-auth-mode]").forEach((button) => {
    const active = button.dataset.authMode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
}

async function submitAuth(event) {
  event.preventDefault();
  const username = $("#username").value.trim();
  const password = $("#password").value;
  const error = $("#authError");
  if (mode === "register" && password !== $("#confirmPassword").value) {
    error.textContent = "两次输入的密码不一致";
    error.hidden = false;
    return;
  }

  $("#authSubmit").disabled = true;
  error.hidden = true;
  try {
    const response = await fetch(`/api/auth/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) throw new Error(data?.detail || `请求失败（HTTP ${response.status}）`);
    window.location.assign(destination());
  } catch (requestError) {
    error.textContent = requestError.message;
    error.hidden = false;
  } finally {
    $("#authSubmit").disabled = false;
  }
}

document.querySelectorAll("[data-auth-mode]").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.authMode));
});
$("#authForm").addEventListener("submit", submitAuth);

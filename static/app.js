const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const messages = document.querySelector("#messages");
const modelSelect = document.querySelector("#modelSelect");
const newChatButton = document.querySelector("#newChatButton");

let history = [];

function appendMessage(role, text, options = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}${options.error ? " error" : ""}`;

  const roleNode = document.createElement("div");
  roleNode.className = "message-role";
  roleNode.textContent = role === "user" ? "Вы" : "Пётр I";

  const textNode = document.createElement("p");
  textNode.textContent = text;

  article.append(roleNode, textNode);

  if (options.sources?.length) {
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = "Опора ответа";
    details.append(summary);

    options.sources.forEach((source, index) => {
      const sourceNode = document.createElement("div");
      sourceNode.className = "source";
      const title = document.createElement("strong");
      title.textContent = `${index + 1}. ${source.source || "Источник"} · score ${source.score}`;
      const body = document.createElement("span");
      body.textContent = source.text;
      sourceNode.append(title, body);
      details.append(sourceNode);
    });

    article.append(details);
  }

  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function resetChat() {
  history = [];
  messages.innerHTML = "";
  appendMessage(
    "assistant",
    "Диалог начат заново. Спрашивай прямо, и я отвечу, как ведаю сам."
  );
}

async function sendMessage(text) {
  const placeholder = appendMessage("assistant", "Думаю над ответом...");
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: text,
      model: modelSelect.value,
      history,
    }),
  });

  const data = await response.json();
  placeholder.remove();

  if (!response.ok) {
    appendMessage("assistant", data.error || "Ошибка запроса.", { error: true });
    return;
  }

  appendMessage("assistant", data.answer, { sources: data.sources || [] });
  history.push({ role: "user", content: text });
  history.push({ role: "assistant", content: data.answer });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  input.value = "";
  input.style.height = "";
  appendMessage("user", text);

  const button = form.querySelector("button");
  button.disabled = true;
  input.disabled = true;
  try {
    await sendMessage(text);
  } finally {
    button.disabled = false;
    input.disabled = false;
    input.focus();
  }
});

modelSelect.addEventListener("change", resetChat);
newChatButton.addEventListener("click", resetChat);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 112)}px`;
});

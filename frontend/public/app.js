/**
 * Corpus frontend — document-scoped RAG, client API key, PDF preview.
 */

(() => {
  "use strict";

  const STORAGE_KEY = "corpus.geminiApiKey";
  const STORAGE_DOC = "corpus.selectedDocId";

  const DOC_STEPS = [
    { id: "upload", name: "Upload" },
    { id: "extraction", name: "Extract" },
    { id: "cleaning", name: "Clean" },
    { id: "chunking", name: "Chunk" },
    { id: "batching", name: "Batch" },
    { id: "embedding", name: "BGE-M3" },
    { id: "storage", name: "ChromaDB" },
  ];

  const QUERY_STEPS = [
    { id: "query_processing", name: "Parse" },
    { id: "query_embedding", name: "Embed" },
    { id: "similarity_search", name: "Search" },
    { id: "context_retrieval", name: "Context" },
    { id: "llm_response", name: "Generate" },
  ];

  const STATUS_LABELS = {
    pending: "Idle",
    processing: "Running",
    completed: "Done",
    error: "Failed",
    streaming: "Streaming",
  };

  const $ = (sel, root = document) => root.querySelector(sel);

  const ui = {
    uploadZone: $("#uploadZone"),
    fileInput: $("#fileInput"),
    documentList: $("#documentList"),
    docEmpty: $("#docEmpty"),
    chatMessages: $("#chatMessages"),
    chatInput: $("#chatInput"),
    chatForm: $("#chatForm"),
    sendBtn: $("#sendBtn"),
    welcomeMsg: $("#welcomeMessage"),
    docPipelineSteps: $("#docPipelineSteps"),
    docPipelineEmpty: $("#docPipelineEmpty"),
    queryPipelineSteps: $("#queryPipelineSteps"),
    queryPipelineEmpty: $("#queryPipelineEmpty"),
    docSectionDot: $("#docSectionDot"),
    querySectionDot: $("#querySectionDot"),
    settingsBtn: $("#settingsBtn"),
    settingsModal: $("#settingsModal"),
    closeSettingsBtn: $("#closeSettingsBtn"),
    keyStatus: $("#keyStatus"),
    apiPill: $("#apiPill"),
    scopePill: $("#scopePill"),
    apiKeyInput: $("#apiKeyInput"),
    saveKeyBtn: $("#saveKeyBtn"),
    clearKeyBtn: $("#clearKeyBtn"),
    toggleKeyBtn: $("#toggleKeyBtn"),
    pdfViewerModal: $("#pdfViewerModal"),
    pdfFrame: $("#pdfFrame"),
    pdfViewerSubtitle: $("#pdfViewerSubtitle"),
    pdfOpenTabBtn: $("#pdfOpenTabBtn"),
    closePdfViewerBtn: $("#closePdfViewerBtn"),
    toastContainer: $("#toastContainer"),
  };

  const state = {
    uploading: false,
    chatting: false,
    timers: {},
    documents: [],
    selectedDocId: localStorage.getItem(STORAGE_DOC) || null,
  };

  /* ── API key helpers ────────────────────────── */

  function getClientApiKey() {
    return (localStorage.getItem(STORAGE_KEY) || "").trim();
  }

  function setClientApiKey(key) {
    const value = (key || "").trim();
    if (value) localStorage.setItem(STORAGE_KEY, value);
    else localStorage.removeItem(STORAGE_KEY);
  }

  function authHeaders(extra = {}) {
    const headers = { ...extra };
    const key = getClientApiKey();
    if (key) headers["X-Gemini-Api-Key"] = key;
    return headers;
  }

  function requireApiKey() {
    if (getClientApiKey()) return true;
    toast("Add your Gemini API key in Settings", "warning");
    openSettings();
    return false;
  }

  /* ── Toasts ─────────────────────────────────── */

  function toast(message, type = "info", duration = 3800) {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = message;
    ui.toastContainer.appendChild(el);
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transition = "opacity 180ms ease";
      setTimeout(() => el.remove(), 200);
    }, duration);
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function truncate(text, n = 28) {
    if (!text) return "";
    return text.length > n ? `${text.slice(0, n - 1)}…` : text;
  }

  /* ── Settings ───────────────────────────────── */

  function refreshApiPill() {
    const key = getClientApiKey();
    if (key) {
      ui.apiPill.textContent = `Key · ${key.slice(0, 6)}…`;
      ui.apiPill.className = "status-pill ok";
      ui.keyStatus.className = "key-status ok";
      ui.keyStatus.textContent = "Client API key is active for this browser.";
    } else {
      ui.apiPill.textContent = "No API key";
      ui.apiPill.className = "status-pill bad";
      ui.keyStatus.className = "key-status bad";
      ui.keyStatus.textContent = "Enter a Gemini API key to chat and upload.";
    }
  }

  function openSettings() {
    ui.apiKeyInput.value = getClientApiKey();
    ui.settingsModal.hidden = false;
    refreshApiPill();
    ui.apiKeyInput.focus();
  }

  function closeSettings() {
    ui.settingsModal.hidden = true;
  }

  function saveApiKey() {
    const value = ui.apiKeyInput.value.trim();
    if (!value) {
      toast("Paste a valid Gemini API key", "warning");
      return;
    }
    setClientApiKey(value);
    refreshApiPill();
    toast("API key saved in this browser", "success");
    closeSettings();
  }

  function clearApiKey() {
    setClientApiKey("");
    ui.apiKeyInput.value = "";
    refreshApiPill();
    toast("API key cleared", "success");
  }

  /* ── Selection / scope ──────────────────────── */

  function selectedDoc() {
    return state.documents.find((d) => d.id === state.selectedDocId) || null;
  }

  function updateScopePill() {
    const doc = selectedDoc();
    if (!doc) {
      ui.scopePill.textContent = "No PDF selected";
      ui.scopePill.className = "status-pill scope-pill";
      ui.chatInput.placeholder = "Select a PDF, then ask a question…";
      return;
    }
    ui.scopePill.textContent = `Asking · ${truncate(doc.name, 32)}`;
    ui.scopePill.className = "status-pill scope-pill active";
    ui.chatInput.placeholder = `Ask about “${truncate(doc.name, 40)}”…`;
  }

  function selectDocument(docId, { persist = true } = {}) {
    state.selectedDocId = docId;
    if (persist) {
      if (docId) localStorage.setItem(STORAGE_DOC, docId);
      else localStorage.removeItem(STORAGE_DOC);
    }
    ui.documentList.querySelectorAll(".doc-item").forEach((el) => {
      el.classList.toggle("selected", el.dataset.id === docId);
    });
    updateScopePill();
  }

  /* ── SSE ────────────────────────────────────── */

  function parseSSEChunk(part, onEvent) {
    if (!part.trim()) return;
    let eventType = "message";
    const dataLines = [];
    for (const line of part.split("\n")) {
      if (line.startsWith("event:")) eventType = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return;
    try {
      onEvent(eventType, JSON.parse(dataLines.join("")));
    } catch (err) {
      console.warn("SSE parse error", err);
    }
  }

  async function fetchSSE(url, options, onEvent) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let message = "Request failed";
      try {
        const body = await response.json();
        message = body.error || body.detail || message;
      } catch (_) {}
      throw new Error(message);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        if (buffer.trim()) parseSSEChunk(buffer.replace(/\r/g, ""), onEvent);
        break;
      }
      buffer += decoder.decode(value, { stream: true }).replace(/\r/g, "");
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) parseSSEChunk(part, onEvent);
    }
  }

  /* ── Pipeline UI ────────────────────────────── */

  function renderSteps(container, steps) {
    container.innerHTML = "";
    steps.forEach((step, index) => {
      const row = document.createElement("div");
      row.className = "step pending";
      row.id = `step-${step.id}`;
      row.innerHTML = `
        <div class="step-dot"></div>
        <div>
          <div class="step-name">${step.name}</div>
          <div class="step-details" id="details-${step.id}">Waiting…</div>
        </div>
        <div class="step-side">
          <span class="step-status" id="status-${step.id}">${STATUS_LABELS.pending}</span>
          <span class="step-time" id="time-${step.id}">—</span>
        </div>
      `;
      container.appendChild(row);
      if (index < steps.length - 1) {
        const connector = document.createElement("div");
        connector.className = "step-connector";
        connector.id = `connector-${step.id}`;
        container.appendChild(connector);
      }
    });
  }

  function updateStep(stepId, status, details) {
    const row = $(`#step-${stepId}`);
    if (!row) return;
    row.className = `step ${status}`;
    const statusEl = $(`#status-${stepId}`);
    const detailsEl = $(`#details-${stepId}`);
    const timeEl = $(`#time-${stepId}`);
    const connector = $(`#connector-${stepId}`);

    if (status === "processing") {
      state.timers[stepId] = Date.now();
      connector?.classList.add("active");
    }
    if (status === "completed" || status === "error") {
      connector?.classList.toggle("completed", status === "completed");
      connector?.classList.remove("active");
      if (state.timers[stepId]) {
        timeEl.textContent = `${((Date.now() - state.timers[stepId]) / 1000).toFixed(1)}s`;
      }
    }
    statusEl.textContent = STATUS_LABELS[status] || status;
    if (details) detailsEl.textContent = details;
  }

  function resetPipeline(containerId, steps, emptyId, dotId) {
    const empty = $(`#${emptyId}`);
    if (empty) empty.style.display = "none";
    $(`#${dotId}`)?.classList.add("active");
    renderSteps($(`#${containerId}`), steps);
    state.timers = {};
  }

  function stopPulse(dotId) {
    $(`#${dotId}`)?.classList.remove("active");
  }

  /* ── Documents ──────────────────────────────── */

  function formatChars(n) {
    if (n > 1_000_000) return `${(n / 1_000_000).toFixed(1)}M chars`;
    if (n > 1_000) return `${(n / 1_000).toFixed(1)}K chars`;
    return `${n} chars`;
  }

  async function loadDocuments() {
    try {
      const res = await fetch("/api/documents");
      const data = await res.json();
      state.documents = data.documents || [];
      if (
        state.selectedDocId &&
        !state.documents.some((d) => d.id === state.selectedDocId)
      ) {
        state.selectedDocId = null;
        localStorage.removeItem(STORAGE_DOC);
      }
      if (!state.selectedDocId && state.documents.length === 1) {
        state.selectedDocId = state.documents[0].id;
        localStorage.setItem(STORAGE_DOC, state.selectedDocId);
      }
      renderDocuments(state.documents);
      updateScopePill();
    } catch (err) {
      console.error(err);
      toast("Could not load documents", "error");
    }
  }

  function renderDocuments(docs) {
    ui.documentList.querySelectorAll(".doc-item").forEach((el) => el.remove());
    if (!docs.length) {
      ui.docEmpty.style.display = "block";
      return;
    }
    ui.docEmpty.style.display = "none";

    docs.forEach((doc) => {
      const item = document.createElement("div");
      item.className = `doc-item${doc.id === state.selectedDocId ? " selected" : ""}`;
      item.dataset.id = doc.id;
      item.innerHTML = `
        <div>
          <div class="doc-name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</div>
          <div class="doc-meta">${doc.pages} pages · ${doc.chunks} chunks · ${formatChars(doc.characters)}</div>
        </div>
        <div class="doc-actions">
          <button type="button" class="doc-action view" data-action="view">View</button>
          <button type="button" class="doc-action delete" data-action="delete">Delete</button>
        </div>
      `;
      item.addEventListener("click", (e) => {
        const action = e.target?.dataset?.action;
        if (action === "view") {
          e.stopPropagation();
          openPdfViewer(doc);
          return;
        }
        if (action === "delete") {
          e.stopPropagation();
          deleteDocument(doc.id);
          return;
        }
        selectDocument(doc.id);
      });
      ui.documentList.appendChild(item);
    });
  }

  async function deleteDocument(docId) {
    if (!confirm("Delete this document and its vectors?")) return;
    try {
      const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || "Delete failed");
      }
      if (state.selectedDocId === docId) selectDocument(null);
      toast("Document removed", "success");
      loadDocuments();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function uploadFile(file) {
    if (state.uploading) {
      toast("Upload already in progress", "warning");
      return;
    }
    if (!requireApiKey()) return;
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
      toast("Please choose a PDF file", "error");
      return;
    }

    state.uploading = true;
    ui.uploadZone.classList.add("disabled");
    resetPipeline("docPipelineSteps", DOC_STEPS, "docPipelineEmpty", "docSectionDot");

    const formData = new FormData();
    formData.append("file", file);

    try {
      let newDocId = null;
      await fetchSSE(
        "/api/upload",
        { method: "POST", headers: authHeaders(), body: formData },
        (type, data) => {
          if (type !== "pipeline") return;
          if (data.step === "complete") {
            newDocId = data.doc_id || null;
            toast(data.details, "success");
            stopPulse("docSectionDot");
          } else {
            updateStep(data.step, data.status, data.details);
          }
        }
      );
      await loadDocuments();
      if (newDocId) selectDocument(newDocId);
    } catch (err) {
      toast(err.message, "error");
    } finally {
      state.uploading = false;
      ui.uploadZone.classList.remove("disabled");
      ui.fileInput.value = "";
    }
  }

  /* ── PDF viewer ─────────────────────────────── */

  function openPdfViewer(doc) {
    const url = `/api/documents/${encodeURIComponent(doc.id)}/file`;
    ui.pdfViewerSubtitle.textContent = doc.name;
    ui.pdfOpenTabBtn.href = url;
    ui.pdfFrame.removeAttribute("srcdoc");
    ui.pdfFrame.src = url;
    ui.pdfViewerModal.hidden = false;

    // Surface API JSON errors inside the iframe instead of a blank/pretty-print page
    ui.pdfFrame.onload = async () => {
      try {
        const res = await fetch(url, { method: "GET" });
        const type = res.headers.get("content-type") || "";
        if (!res.ok || !type.includes("pdf")) {
          const body = await res.json().catch(() => ({}));
          const message = body.error || body.detail || `Unable to open PDF (${res.status})`;
          ui.pdfFrame.removeAttribute("src");
          ui.pdfFrame.srcdoc = `
            <html><body style="font-family:sans-serif;background:#111;color:#eee;padding:24px">
              <h3 style="margin:0 0 8px">Preview unavailable</h3>
              <p style="opacity:.85">${String(message).replace(/[<>&]/g, "")}</p>
            </body></html>`;
          toast(message, "error");
        }
      } catch (err) {
        toast(err.message || "Could not load PDF", "error");
      }
    };
  }

  function closePdfViewer() {
    ui.pdfViewerModal.hidden = true;
    ui.pdfFrame.src = "about:blank";
  }

  /* ── Markdown ───────────────────────────────── */

  function renderMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
      `<pre><code class="language-${lang}">${code.trim()}</code></pre>`
    );
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");
    html = html.replace(/^[*-] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");
    html = html.replace(/\n\n/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");
    return `<p>${html}</p>`
      .replace(/<p>\s*<\/p>/g, "")
      .replace(/<p>\s*(<h[1-3]>)/g, "$1")
      .replace(/(<\/h[1-3]>)\s*<\/p>/g, "$1")
      .replace(/<p>\s*(<pre>)/g, "$1")
      .replace(/(<\/pre>)\s*<\/p>/g, "$1")
      .replace(/<p>\s*(<ul>)/g, "$1")
      .replace(/(<\/ul>)\s*<\/p>/g, "$1");
  }

  /* ── Chat ───────────────────────────────────── */

  function hideWelcome() {
    if (ui.welcomeMsg) ui.welcomeMsg.style.display = "none";
  }

  function scrollChat() {
    ui.chatMessages.scrollTop = ui.chatMessages.scrollHeight;
  }

  function addUserMessage(text) {
    hideWelcome();
    const el = document.createElement("div");
    el.className = "message user";
    el.innerHTML = `
      <div class="message-avatar">YOU</div>
      <div class="message-body">${renderMarkdown(text)}</div>
    `;
    ui.chatMessages.appendChild(el);
    scrollChat();
  }

  function addAssistantShell() {
    hideWelcome();
    const el = document.createElement("div");
    el.className = "message assistant";
    el.id = "current-response";
    el.innerHTML = `
      <div class="message-avatar">AI</div>
      <div class="message-body">
        <div class="typing" aria-label="Generating"><i></i><i></i><i></i></div>
      </div>
    `;
    ui.chatMessages.appendChild(el);
    scrollChat();
    return el;
  }

  function appendToken(token) {
    const el = $("#current-response");
    if (!el) return;
    const body = el.querySelector(".message-body");
    body.querySelector(".typing")?.remove();
    const raw = (body.getAttribute("data-raw") || "") + token;
    body.setAttribute("data-raw", raw);
    body.innerHTML = renderMarkdown(raw);
    scrollChat();
  }

  function finalizeAssistant(sources) {
    const el = $("#current-response");
    if (!el) return;
    const body = el.querySelector(".message-body");
    body.querySelector(".typing")?.remove();
    if (sources?.length) {
      const wrap = document.createElement("div");
      wrap.className = "message-sources";
      wrap.innerHTML = sources
        .map((s) => `<span class="source-tag">${escapeHtml(s)}</span>`)
        .join("");
      body.appendChild(wrap);
    }
    el.removeAttribute("id");
    scrollChat();
  }

  function addErrorMessage(text) {
    hideWelcome();
    const el = document.createElement("div");
    el.className = "message assistant";
    el.innerHTML = `
      <div class="message-avatar">AI</div>
      <div class="message-body error"><strong>Error:</strong> ${escapeHtml(text)}</div>
    `;
    ui.chatMessages.appendChild(el);
    scrollChat();
  }

  async function sendMessage(event) {
    event?.preventDefault();
    const question = ui.chatInput.value.trim();
    if (!question || state.chatting) return;

    if (!requireApiKey()) return;
    if (!state.selectedDocId) {
      toast("Select a PDF from the library first", "warning");
      return;
    }

    state.chatting = true;
    ui.chatInput.value = "";
    ui.sendBtn.disabled = true;

    addUserMessage(question);
    const shell = addAssistantShell();
    resetPipeline("queryPipelineSteps", QUERY_STEPS, "queryPipelineEmpty", "querySectionDot");

    try {
      await fetchSSE(
        "/api/chat",
        {
          method: "POST",
          headers: authHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify({
            question,
            doc_id: state.selectedDocId,
          }),
        },
        (type, data) => {
          if (type !== "pipeline") return;
          if (data.step === "complete") {
            finalizeAssistant(data.sources);
            stopPulse("querySectionDot");
          } else if (data.status === "streaming") {
            updateStep(data.step, "streaming", "Streaming tokens…");
            appendToken(data.token);
          } else {
            updateStep(data.step, data.status, data.details);
            if (data.status === "error") {
              shell.remove();
              addErrorMessage(data.details);
              stopPulse("querySectionDot");
            }
          }
        }
      );
    } catch (err) {
      shell.remove();
      addErrorMessage(err.message);
    } finally {
      state.chatting = false;
      ui.sendBtn.disabled = false;
      ui.chatInput.focus();
    }
  }

  /* ── Events ─────────────────────────────────── */

  ui.settingsBtn.addEventListener("click", openSettings);
  ui.closeSettingsBtn.addEventListener("click", closeSettings);
  ui.saveKeyBtn.addEventListener("click", saveApiKey);
  ui.clearKeyBtn.addEventListener("click", clearApiKey);
  ui.toggleKeyBtn.addEventListener("click", () => {
    const isPassword = ui.apiKeyInput.type === "password";
    ui.apiKeyInput.type = isPassword ? "text" : "password";
    ui.toggleKeyBtn.textContent = isPassword ? "Hide" : "Show";
  });
  ui.settingsModal.addEventListener("click", (e) => {
    if (e.target === ui.settingsModal) closeSettings();
  });

  ui.closePdfViewerBtn.addEventListener("click", closePdfViewer);
  ui.pdfViewerModal.addEventListener("click", (e) => {
    if (e.target === ui.pdfViewerModal) closePdfViewer();
  });

  ui.uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    ui.uploadZone.classList.add("drag-over");
  });
  ui.uploadZone.addEventListener("dragleave", () => ui.uploadZone.classList.remove("drag-over"));
  ui.uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    ui.uploadZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
  ui.uploadZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      ui.fileInput.click();
    }
  });
  ui.fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) uploadFile(file);
  });

  ui.chatForm.addEventListener("submit", sendMessage);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!ui.pdfViewerModal.hidden) closePdfViewer();
      else if (!ui.settingsModal.hidden) closeSettings();
    }
  });

  document.addEventListener("DOMContentLoaded", () => {
    refreshApiPill();
    loadDocuments();
    ui.chatInput.focus();
  });
})();

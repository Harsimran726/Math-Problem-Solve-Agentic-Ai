/**
 * Math Solver Agent — Frontend JS
 * Handles: image drag-drop, form submission, polling, HITL, Markdown + KaTeX rendering, logs.
 */
(function () {
  "use strict";

  // ── DOM refs ────────────────────────────────────────────────────────
  const form = document.getElementById("solve-form");
  const textInput = document.getElementById("text_input");
  const imageFileInput = document.getElementById("image_file");
  const dropzone = document.getElementById("dropzone");
  const dropzoneContent = document.getElementById("dropzone-content");
  const imagePreview = document.getElementById("image-preview");
  const submitBtn = document.getElementById("submit-btn");

  const stepperCard = document.getElementById("stepper-card");
  const hitlCard = document.getElementById("hitl-card");
  const hitlContent = document.getElementById("hitl-content");
  const btnProceed = document.getElementById("btn-proceed");
  const btnRetry = document.getElementById("btn-retry");

  const resultsCard = document.getElementById("results-card");
  const resultProblem = document.getElementById("result-problem");
  const resultSolution = document.getElementById("result-solution");
  const resultContext = document.getElementById("result-context");
  const verifyBadge = document.getElementById("verify-badge");
  const verifyScore = document.getElementById("verify-score");

  const logsCard = document.getElementById("logs-card");
  const logsToggle = document.getElementById("logs-toggle");
  const logsBody = document.getElementById("logs-body");
  const logsList = document.getElementById("logs-list");
  const logsCount = document.getElementById("logs-count");
  const logsChevron = document.getElementById("logs-chevron");

  let currentSessionId = null;
  let pollTimer = null;

  // Step order for the stepper
  const STEP_ORDER = ["parser", "solver", "verifier", "explanation"];

  // ── Image Drag & Drop ──────────────────────────────────────────────

  dropzone.addEventListener("click", () => imageFileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const dt = new DataTransfer();
      dt.items.add(file);
      imageFileInput.files = dt.files;
      showPreview(file);
    }
  });

  imageFileInput.addEventListener("change", () => {
    if (imageFileInput.files[0]) {
      showPreview(imageFileInput.files[0]);
    }
  });

  function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      imagePreview.classList.add("visible");
      dropzoneContent.style.display = "none";
    };
    reader.readAsDataURL(file);
  }

  // ── Form submission ────────────────────────────────────────────────

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const text = textInput.value.trim();
    const file = imageFileInput.files[0];

    if (!text && !file) {
      alert("Please enter a question or upload an image.");
      return;
    }

    resetResults();
    setLoading(true);
    stepperCard.hidden = false;
    logsCard.hidden = false;
    updateStepper("parser");

    const fd = new FormData();
    fd.append("text_input", text);
    if (file) fd.append("image_file", file);

    try {
      const res = await fetch("/solve", { method: "POST", body: fd });
      const data = await res.json();
      currentSessionId = data.session_id;
      startPolling();
    } catch (err) {
      console.error("Solve error:", err);
      setLoading(false);
      alert("Failed to start solving. Check the console for details.");
    }
  });

  // ── Polling ────────────────────────────────────────────────────────

  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollStatus, 2000);
    pollStatus();
  }

  async function pollStatus() {
    if (!currentSessionId) return;

    try {
      const res = await fetch(`/status/${currentSessionId}`);
      const data = await res.json();

      if (data.logs && data.logs.length > 0) {
        updateLogs(data.logs);
      }

      if (data.current_step && data.current_step !== "done" && data.current_step !== "awaiting_clarification") {
        updateStepper(data.current_step);
      }

      if (data.status === "need_clarification") {
        clearInterval(pollTimer);
        setLoading(false);
        updateStepper("parser", true);
        showHITL(data.state);
        return;
      }

      if (data.status === "completed") {
        clearInterval(pollTimer);
        setLoading(false);
        completeStepper();
        showResults(data.state);
        return;
      }

      if (data.status === "error") {
        clearInterval(pollTimer);
        setLoading(false);
        alert("An error occurred: " + (data.error || "Unknown error"));
        return;
      }
    } catch (err) {
      console.error("Poll error:", err);
    }
  }

  // ── HITL Clarification ─────────────────────────────────────────────

  function showHITL(state) {
    hitlCard.hidden = false;
    const parse = state.parse_agent_output || {};

    let html = "";
    if (parse.topic) html += `<p><strong>Topic:</strong> ${escapeHtml(parse.topic)}</p>`;
    if (parse.problem_text) html += `<p><strong>Problem:</strong> ${escapeHtml(parse.problem_text)}</p>`;
    if (parse.constraints && parse.constraints.length) {
      html += `<p><strong>Constraints:</strong> ${parse.constraints.map(escapeHtml).join(", ")}</p>`;
    }
    if (parse.variables && parse.variables.length) {
      html += `<p><strong>Variables:</strong> ${parse.variables.map(escapeHtml).join(", ")}</p>`;
    }
    hitlContent.innerHTML = html || "<p>The parser is unsure and needs your confirmation.</p>";
    hitlCard.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  btnProceed.addEventListener("click", async () => {
    hitlCard.hidden = true;
    setLoading(true);
    updateStepper("solver");
    try {
      await fetch(`/proceed/${currentSessionId}`, { method: "POST" });
      startPolling();
    } catch (err) {
      console.error("Proceed error:", err);
      setLoading(false);
    }
  });

  btnRetry.addEventListener("click", async () => {
    hitlCard.hidden = true;
    setLoading(true);
    updateStepper("parser");
    try {
      await fetch(`/retry/${currentSessionId}`, { method: "POST" });
      startPolling();
    } catch (err) {
      console.error("Retry error:", err);
      setLoading(false);
    }
  });

  // ── Stepper ────────────────────────────────────────────────────────

  function updateStepper(currentStep, waitingHITL) {
    const steps = document.querySelectorAll(".stepper__step");
    const lines = document.querySelectorAll(".stepper__line");
    const idx = STEP_ORDER.indexOf(currentStep);

    steps.forEach((el, i) => {
      el.classList.remove("active", "done");
      if (i < idx) el.classList.add("done");
      else if (i === idx) el.classList.add("active");
    });

    lines.forEach((el, i) => {
      el.classList.remove("done");
      if (i < idx) el.classList.add("done");
    });
  }

  function completeStepper() {
    document.querySelectorAll(".stepper__step").forEach((el) => {
      el.classList.remove("active");
      el.classList.add("done");
    });
    document.querySelectorAll(".stepper__line").forEach((el) => {
      el.classList.add("done");
    });
  }

  // ── Results (Explanation Only + Verification Badge) ─────────────────

  function showResults(state) {
    resultsCard.hidden = false;

    const verify = state.verify_agent_output || {};
    const explain = state.explain_agent_output || {};

    // Verification confidence badge
    if (verify.confidence_score != null) {
      const pct = verify.confidence_score;
      verifyBadge.hidden = false;
      verifyBadge.className = "verify-badge " + (pct >= 80 ? "high" : pct >= 50 ? "medium" : "low");
      verifyScore.textContent = `Confidence: ${pct}%`;
    }

    // Render explanation sections with Markdown + KaTeX
    resultProblem.innerHTML = renderMathMarkdown(explain.problem_statement || "");
    resultSolution.innerHTML = renderMathMarkdown(explain.solution || "");
    resultContext.innerHTML = renderMathMarkdown(explain.context || "");

    // Render KaTeX in all math-content blocks
    renderKaTeX();

    resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ── Logs ───────────────────────────────────────────────────────────

  function updateLogs(logs) {
    logsList.innerHTML = "";
    logs.forEach((entry) => {
      const li = document.createElement("li");
      li.textContent = entry;
      logsList.appendChild(li);
    });
    logsCount.textContent = logs.length;
  }

  logsToggle.addEventListener("click", () => {
    const isOpen = !logsBody.hidden;
    logsBody.hidden = isOpen;
    logsChevron.classList.toggle("open", !isOpen);
  });

  // ── Helpers ────────────────────────────────────────────────────────

  function setLoading(loading) {
    submitBtn.disabled = loading;
    submitBtn.classList.toggle("loading", loading);
  }

  function resetResults() {
    hitlCard.hidden = true;
    resultsCard.hidden = true;
    verifyBadge.hidden = true;
    resultProblem.innerHTML = "";
    resultSolution.innerHTML = "";
    resultContext.innerHTML = "";
    logsList.innerHTML = "";
    logsCount.textContent = "0";
  }

  function renderMathMarkdown(text) {
    if (!text) return "";
    // First render Markdown
    let html = "";
    if (typeof marked !== "undefined" && marked.parse) {
      html = marked.parse(String(text));
    } else {
      html = `<pre>${escapeHtml(String(text))}</pre>`;
    }
    return html;
  }

  function renderKaTeX() {
    // Use KaTeX auto-render on all math-content elements
    if (typeof renderMathInElement === "undefined") return;
    document.querySelectorAll(".math-content").forEach((el) => {
      renderMathInElement(el, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
        throwOnError: false,
      });
    });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
})();

const $ = (id) => document.getElementById(id);
let models = [];
let busy = false;
let output = "";

function status(text, kind = "") {
  $("status").textContent = text;
  $("status").className = `status ${kind}`;
}

async function request(path, data) {
  const response = await fetch(path, data === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "The request failed. Please try again.");
  return result;
}

function current() {
  return models.find((model) => model.label === $("model").value) || models[0];
}

function setBusy(value) {
  busy = value;
  for (const id of ["generate", "query", "prompt", "reset-prompt", "model", "known"]) $(id).disabled = value;
  $("query-form").setAttribute("aria-busy", String(value));
}

function clearResult() {
  output = "";
  $("result-content").hidden = true;
  $("result-empty").hidden = false;
  $("result-badge").textContent = "OUTPUT";
  $("result-badge").className = "small-tag";
  $("result-footer-text").textContent = "Schema-compiled. Strictly compared when a record exists.";
  $("gold").replaceChildren();
  $("gold-empty").hidden = false;
  $("verdict").textContent = "NO RECORD";
  $("verdict").className = "count-pill";
}

// Highlight using text nodes: catalog and model content never become HTML.
function showJSON(element, text) {
  element.replaceChildren();
  const tokens = /"(?:\\.|[^"\\])*"\s*:|"(?:\\.|[^"\\])*"|\b(?:true|false|null|\d+(?:\.\d+)?)\b/g;
  let last = 0;
  for (const match of text.matchAll(tokens)) {
    element.append(document.createTextNode(text.slice(last, match.index)));
    const span = document.createElement("span");
    span.className = match[0].endsWith(":") ? "json-key" : match[0].startsWith('"') ? "json-string" : "json-number";
    span.textContent = match[0];
    element.append(span);
    last = match.index + match[0].length;
  }
  element.append(document.createTextNode(text.slice(last)));
}

function renderGold(query, result) {
  if (result.expected === null || result.expected === undefined) return;
  const card = $("gold-template").content.cloneNode(true);
  card.querySelector(".similarity").textContent = result.exact ? "strict comparator: match" : "strict comparator: mismatch";
  card.querySelector(".example-text").textContent = query;
  showJSON(card.querySelector("pre"), JSON.stringify(result.expected, null, 2));
  for (const layer of result.expected.layers || []) {
    const chip = document.createElement("span");
    chip.className = "layer-chip";
    chip.textContent = layer;
    card.querySelector(".example-layers").append(chip);
  }
  $("gold").replaceChildren(card);
  $("gold-empty").hidden = true;
  $("verdict").textContent = result.exact ? "EXACT MATCH" : "MISMATCH";
  $("verdict").className = `count-pill verdict ${result.exact ? "match" : "mismatch"}`;
}

async function run() {
  if (busy || !$("query").reportValidity()) return;
  const query = $("query").value.trim();
  if (!query) return status("Enter a question to get started.", "error");
  const model = current();
  const prefix = $("prompt").value;
  setBusy(true);
  clearResult();
  $("result-badge").textContent = "GENERATING";
  status(`Decoding with ${model.label}… The first question loads the model.`, "busy");
  try {
    const body = { query, model: model.label };
    if (prefix !== model.prompt_prefix) body.prompt_prefix = prefix;
    const result = await request("/api/generate", body);
    $("result-empty").hidden = true;
    $("result-content").hidden = false;
    if (!result.valid) {
      output = result.error;
      $("result-code").textContent = result.error;
      $("result-badge").textContent = "INVALID OUTPUT";
      $("result-badge").className = "small-tag invalid";
      $("result-description").textContent = "DECODING ERROR";
      $("copy").textContent = "Copy message";
      $("result-summary").textContent = "The model could not complete a schema-valid FELN within its output budget.";
      status("Decoding stopped before a valid FELN was produced.", "error");
      return;
    }
    output = JSON.stringify(result.feln, null, 2);
    showJSON($("result-code"), output);
    $("result-badge").textContent = "VALID FELN";
    $("result-badge").className = "small-tag valid";
    $("result-description").textContent = "FELN / JSON · schema-compiled";
    $("copy").textContent = "Copy JSON";
    $("result-summary").textContent = `${result.feln.layers.join(" → ")} · ${result.feln.relations.join(" · ") || "No spatial join"}`;
    $("result-footer-text").textContent = `${result.generated_tokens} tokens in ${result.seconds.toFixed(2)} s · first token ${result.first_token_seconds.toFixed(2)} s · ${result.prompt_tokens} prompt tokens · ${result.backend}`;
    renderGold(query, result);
    status(result.exact === null || result.exact === undefined
      ? "Your FELN is ready. Validated against the schema grammar."
      : result.exact ? "Your FELN matches the recorded answer." : "Your FELN differs from the recorded answer. Compare the two.",
      result.exact === false ? "error" : "");
  } catch (error) {
    $("result-badge").textContent = "TRY AGAIN";
    status(error.message || "Could not connect to the server. Please try again.", "error");
  } finally {
    setBusy(false);
  }
}

$("query-form").addEventListener("submit", (event) => { event.preventDefault(); run(); });
$("query").addEventListener("input", () => {
  $("character-count").textContent = `${$("query").value.length} characters`;
  if ($("known").value !== $("query").value) $("known").value = "";
  clearResult();
  status("Query changed. Generate a new FELN.");
});
$("known").addEventListener("change", () => {
  if (!$("known").value) return;
  $("query").value = $("known").value;
  $("query").dispatchEvent(new Event("input"));
  $("known").value = $("query").value;
});
$("model").addEventListener("change", () => {
  $("prompt").value = current().prompt_prefix;
  $("bundle").textContent = current().bundle;
  clearResult();
  status(`Switched to ${current().label}. Generate a new FELN.`);
});
$("prompt").addEventListener("input", () => {
  clearResult();
  status("Prompt changed. This is off-distribution for a fine-tuned model; generate to see the effect.");
});
$("reset-prompt").addEventListener("click", () => {
  $("prompt").value = current().prompt_prefix;
  $("prompt").dispatchEvent(new Event("input"));
});
$("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(output);
    $("copy").textContent = "Copied!";
  } catch {
    status("Clipboard unavailable. Select and copy the output directly.", "error");
  }
});

async function init() {
  $("character-count").textContent = `${$("query").value.length} characters`;
  try {
    const config = await request("/api/config");
    models = config.models;
    $("model").replaceChildren(...models.map((model) => new Option(model.label, model.label)));
    $("known").append(...config.questions.map((question) => new Option(question, question)));
    $("prompt").value = current().prompt_prefix;
    $("bundle").textContent = current().bundle;
    $("corpus-info").textContent = `${models.length} model${models.length === 1 ? "" : "s"} · ${config.questions.length} recorded questions`;
    setBusy(false);
    status("Workspace ready. Ask the question above, or pick a recorded one.");
  } catch {
    $("bundle").textContent = "Disconnected";
    status("Could not connect to the local server. Start python -m src.studio and reload this page.", "error");
  }
}
init();

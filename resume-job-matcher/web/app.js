const messages = document.getElementById('messages');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const statusLine = document.getElementById('statusLine');

const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const docType = document.getElementById('docType');
const uploadStatus = document.getElementById('uploadStatus');
const modelSelect = document.getElementById('modelSelect');
const quickActions = document.getElementById('quickActions');
const quickHelp = document.getElementById('quickHelp');
const searchHistory = document.getElementById('searchHistory');
const historyHelp = document.getElementById('historyHelp');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const favoritePrompts = document.getElementById('favoritePrompts');
const favoriteHelp = document.getElementById('favoriteHelp');

let recentResumeIds = [];
let recentJobIds = [];
let queryHistory = [];
let favoriteQueries = [];

const HISTORY_KEY = 'resume_job_agent_query_history_v1';
const FAVORITES_KEY = 'resume_job_agent_favorite_queries_v1';

function addMessage(role, text) {
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  el.textContent = text;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

function mergeRecent(existing, incoming, limit = 4) {
  const map = new Set(existing);
  for (const id of incoming || []) {
    if (!id || map.has(id)) continue;
    existing.unshift(id);
    map.add(id);
  }
  return existing.slice(0, limit);
}

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (Array.isArray(parsed)) queryHistory = parsed;
  } catch {
    queryHistory = [];
  }
}

function loadFavorites() {
  try {
    const raw = localStorage.getItem(FAVORITES_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (Array.isArray(parsed)) favoriteQueries = parsed;
  } catch {
    favoriteQueries = [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(queryHistory.slice(0, 12)));
  } catch {
    // ignore storage errors
  }
}

function saveFavorites() {
  try {
    localStorage.setItem(FAVORITES_KEY, JSON.stringify(favoriteQueries.slice(0, 12)));
  } catch {
    // ignore storage errors
  }
}

function addToHistory(query) {
  const q = (query || '').trim();
  if (!q) return;
  queryHistory = queryHistory.filter(item => item.toLowerCase() !== q.toLowerCase());
  queryHistory.unshift(q);
  queryHistory = queryHistory.slice(0, 12);
  saveHistory();
  renderSearchHistory();
}

function isFavorite(query) {
  return favoriteQueries.some(item => item.toLowerCase() === query.toLowerCase());
}

function toggleFavorite(query) {
  const q = (query || '').trim();
  if (!q) return;
  if (isFavorite(q)) {
    favoriteQueries = favoriteQueries.filter(item => item.toLowerCase() !== q.toLowerCase());
  } else {
    favoriteQueries.unshift(q);
    favoriteQueries = favoriteQueries.slice(0, 12);
  }
  saveFavorites();
  renderFavorites();
  renderSearchHistory();
}

function clearHistory() {
  queryHistory = [];
  saveHistory();
  renderSearchHistory();
}

function renderSearchHistory() {
  if (!searchHistory || !historyHelp) return;
  searchHistory.innerHTML = '';
  if (!queryHistory.length) {
    historyHelp.textContent = 'No search history yet.';
    return;
  }

  historyHelp.textContent = 'Click a previous query to run again.';
  for (const query of queryHistory.slice(0, 8)) {
    const row = document.createElement('div');
    row.className = 'history-row';

    const runBtn = document.createElement('button');
    runBtn.type = 'button';
    runBtn.className = 'history-item';
    runBtn.textContent = query;
    runBtn.addEventListener('click', () => sendMessage(query));

    const pinBtn = document.createElement('button');
    pinBtn.type = 'button';
    pinBtn.className = 'mini-btn';
    pinBtn.textContent = isFavorite(query) ? '★' : '☆';
    pinBtn.title = isFavorite(query) ? 'Unpin prompt' : 'Pin prompt';
    pinBtn.addEventListener('click', () => toggleFavorite(query));

    row.appendChild(runBtn);
    row.appendChild(pinBtn);
    searchHistory.appendChild(row);
  }
}

function renderFavorites() {
  if (!favoritePrompts || !favoriteHelp) return;
  favoritePrompts.innerHTML = '';

  if (!favoriteQueries.length) {
    favoriteHelp.textContent = 'No pinned prompts yet.';
    return;
  }

  favoriteHelp.textContent = 'Pinned prompts (click to run).';
  for (const query of favoriteQueries.slice(0, 8)) {
    const row = document.createElement('div');
    row.className = 'history-row';

    const runBtn = document.createElement('button');
    runBtn.type = 'button';
    runBtn.className = 'history-item';
    runBtn.textContent = query;
    runBtn.addEventListener('click', () => sendMessage(query));

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'mini-btn';
    removeBtn.textContent = '✕';
    removeBtn.title = 'Remove pinned prompt';
    removeBtn.addEventListener('click', () => toggleFavorite(query));

    row.appendChild(runBtn);
    row.appendChild(removeBtn);
    favoritePrompts.appendChild(row);
  }
}

function renderQuickActions() {
  if (!quickActions || !quickHelp) return;
  quickActions.innerHTML = '';

  const items = [];
  for (const rid of recentResumeIds) {
    items.push({ label: `Find jobs: ${rid}`, text: `find jobs for ${rid}` });
    items.push({ label: `Improve: ${rid}`, text: `improve resume ${rid}` });
  }
  for (const jid of recentJobIds) {
    items.push({ label: `Find candidates: ${jid}`, text: `find candidates for ${jid}` });
  }

  if (!items.length) {
    quickHelp.textContent = 'No recent uploads yet.';
    return;
  }

  quickHelp.textContent = 'Click to run commands quickly.';
  for (const item of items.slice(0, 8)) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'chip';
    btn.textContent = item.label;
    btn.addEventListener('click', () => sendMessage(item.text));
    quickActions.appendChild(btn);
  }
}

async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    statusLine.textContent = `Endee: ${data.endee ? 'OK' : 'Down'} | Ollama: ${data.ollama ? 'OK' : 'Off'} | Indexes: ${data.indexes.join(', ') || 'none'}`;
  } catch {
    statusLine.textContent = 'Unable to connect to backend';
  }
}

async function loadModels() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    if (!data.ollama) return;

    for (const model of data.models || []) {
      if ((model || '').toLowerCase().includes('tinyllama')) continue;
      const opt = document.createElement('option');
      opt.value = model;
      opt.textContent = model;
      modelSelect.appendChild(opt);
    }

    if (data.recommended) {
      modelSelect.value = data.recommended;
    }
  } catch {
    // keep default auto mode
  }
}

async function sendMessage(rawText) {
  const text = (rawText || '').trim();
  if (!text) return;

  addMessage('user', text);
  addToHistory(text);
  messageInput.value = '';
  sendBtn.disabled = true;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        model: modelSelect?.value || null,
      })
    });
    const data = await res.json();
    if (!res.ok) {
      addMessage('assistant', `Error: ${data.detail || 'request failed'}`);
    } else {
      addMessage('assistant', data.reply || '(no response)');
    }
  } catch (err) {
    addMessage('assistant', `Error: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
    messageInput.focus();
  }
}

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  await sendMessage(messageInput.value);
});

if (clearHistoryBtn) {
  clearHistoryBtn.addEventListener('click', clearHistory);
}

uploadBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) {
    uploadStatus.textContent = 'Choose a JSON, PDF, or DOCX file first.';
    return;
  }

  uploadBtn.disabled = true;
  uploadStatus.textContent = 'Uploading...';

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch(`/api/upload?doc_type=${encodeURIComponent(docType.value)}`, {
      method: 'POST',
      body: form,
    });
    const data = await res.json();
    if (!res.ok) {
      uploadStatus.textContent = `Error: ${data.detail || 'upload failed'}`;
    } else {
      uploadStatus.textContent = data.message;
      addMessage('assistant', `Upload result:\n${data.message}`);
      recentResumeIds = mergeRecent(recentResumeIds, data.uploaded_resume_ids || []);
      recentJobIds = mergeRecent(recentJobIds, data.uploaded_job_ids || []);
      renderQuickActions();
      checkStatus();
    }
  } catch (err) {
    uploadStatus.textContent = `Error: ${err.message}`;
  } finally {
    uploadBtn.disabled = false;
  }
});

addMessage('assistant', 'Hi! I can match resumes and jobs, explain matches, and analyze improvements. Upload JSON data or ask a question.');
loadHistory();
loadFavorites();
checkStatus();
loadModels();
renderQuickActions();
renderSearchHistory();
renderFavorites();

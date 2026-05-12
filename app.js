// Spext PWA — minimal MVP.
// Whisper Tiny in browser via transformers.js, IndexedDB for notes.

import {
  pipeline,
  env
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';

// Don't allow remote model fetch beyond the first download — uses HF CDN once, then cached.
env.allowLocalModels = false;
env.useBrowserCache = true;

// ---- DOM ----
const statusEl = document.getElementById('status');
const micBtn = document.getElementById('micBtn');
const micLabel = document.getElementById('micLabel');
const transcriptEl = document.getElementById('transcript');
const detectedLangEl = document.getElementById('detectedLang');
const saveBtn = document.getElementById('saveBtn');
const notesToggleBtn = document.getElementById('notesToggleBtn');
const recordView = document.getElementById('recordView');
const notesView = document.getElementById('notesView');
const detailView = document.getElementById('detailView');
const settingsView = document.getElementById('settingsView');
const settingsBtn = document.getElementById('settingsBtn');
const settingsBackBtn = document.getElementById('settingsBackBtn');
const exportBtn = document.getElementById('exportBtn');
const exportStatus = document.getElementById('exportStatus');
const noteCountText = document.getElementById('noteCountText');
const licensesBtn = document.getElementById('licensesBtn');
const licensesDialog = document.getElementById('licensesDialog');
const licensesCloseBtn = document.getElementById('licensesCloseBtn');
const installBtn = document.getElementById('installBtn');
const backBtn = document.getElementById('backBtn');
const detailBackBtn = document.getElementById('detailBackBtn');
const notesList = document.getElementById('notesList');
const emptyState = document.getElementById('emptyState');
const searchInput = document.getElementById('searchInput');
const detailTitle = document.getElementById('detailTitle');
const detailContent = document.getElementById('detailContent');
const shareBtn = document.getElementById('shareBtn');
const downloadBtn = document.getElementById('downloadBtn');
const deleteBtn = document.getElementById('deleteBtn');

// ---- State ----
let transcriber = null;
let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let rawTranscript = '';
let detectedLanguage = '';
let currentNoteId = null;
let selectedLanguage = 'auto'; // 'auto', 'german', 'english'

// Language picker behavior.
document.querySelectorAll('.lang-option').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.lang-option').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedLanguage = btn.dataset.lang;
  });
});

// ---- IndexedDB ----
const DB_NAME = 'spext';
const DB_VERSION = 1;
const STORE = 'notes';

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
        store.createIndex('updatedAt', 'updatedAt');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbAll() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).getAll();
    req.onsuccess = () => resolve(req.result.sort((a, b) => b.updatedAt - a.updatedAt));
    req.onerror = () => reject(req.error);
  });
}

async function dbSave(note) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    const req = note.id ? store.put(note) : store.add(note);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbDelete(id) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    const req = tx.objectStore(STORE).delete(id);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

async function dbGet(id) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(id);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// ---- Whisper load ----
// Large-v3 Turbo: highest quality multilingual model that fits in ~1.5 GB.
// Picked for desktop-first use (Mac + Windows browsers have plenty of RAM).
const WHISPER_MODEL = 'onnx-community/whisper-large-v3-turbo';

async function loadWhisper() {
  // Detect WebGPU. Available in Chrome/Edge on Mac M-series and most Windows GPUs.
  // Falls back to CPU/WASM if not supported (Safari, older browsers).
  const useWebGPU = !!navigator.gpu;
  const device = useWebGPU ? 'webgpu' : 'wasm';
  const dtype = useWebGPU ? 'fp16' : 'q8';

  try {
    statusEl.textContent = `Loading Whisper Large-v3 Turbo (~1.5 GB, one-time) — ${device.toUpperCase()}…`;
    transcriber = await pipeline(
      'automatic-speech-recognition',
      WHISPER_MODEL,
      {
        device,
        dtype,
        progress_callback: (p) => {
          if (p.status === 'progress' && p.file?.endsWith('.onnx')) {
            const pct = Math.round((p.loaded / p.total) * 100);
            statusEl.textContent = `Loading Whisper… ${pct}% (${device.toUpperCase()})`;
          }
        }
      }
    );
    statusEl.textContent = `Whisper ready. ${useWebGPU ? '⚡ GPU accelerated.' : 'CPU mode.'}`;
    micBtn.disabled = false;
    micLabel.textContent = 'Tap to record';
  } catch (err) {
    statusEl.textContent = `Failed to load model: ${err.message}`;
    console.error(err);
  }
}

// ---- Recording ----
async function startRecording() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(mediaStream);
    audioChunks = [];
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };
    mediaRecorder.onstop = onRecordingStopped;
    mediaRecorder.start();
    isRecording = true;
    micBtn.classList.add('recording');
    document.getElementById('micIcon').textContent = '⏹';
    micLabel.textContent = 'Recording… tap to stop';
    rawTranscript = '';
    transcriptEl.textContent = '';
  } catch (err) {
    statusEl.textContent = `Mic error: ${err.message}`;
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  mediaStream?.getTracks().forEach(t => t.stop());
  isRecording = false;
  micBtn.classList.remove('recording');
  document.getElementById('micIcon').textContent = '🎤';
  micLabel.textContent = 'Transcribing…';
  micBtn.disabled = true;
}

async function onRecordingStopped() {
  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  const arrayBuffer = await blob.arrayBuffer();

  // Decode to PCM Float32 at 16kHz (Whisper expects this).
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const float32 = audioBuffer.getChannelData(0);

  // Reset.
  rawTranscript = '';
  detectedLanguage = '';
  detectedLangEl.hidden = true;

  try {
    // Transcribe in the selected language. "auto" lets Whisper detect; explicit values force it.
    const options = { task: 'transcribe', return_language: true };
    if (selectedLanguage !== 'auto') options.language = selectedLanguage;
    const result = await transcriber(float32, options);
    rawTranscript = (result.text || '').trim();
    detectedLanguage = result.language || (selectedLanguage !== 'auto' ? selectedLanguage : '');
    transcriptEl.textContent = rawTranscript || '(no speech detected)';
    if (detectedLanguage) {
      detectedLangEl.textContent = detectedLanguage;
      detectedLangEl.hidden = false;
    }
    saveBtn.disabled = rawTranscript.length === 0;
  } catch (err) {
    transcriptEl.textContent = `Transcription failed: ${err.message}`;
  } finally {
    micBtn.disabled = false;
    micLabel.textContent = 'Tap to record';
  }
}

micBtn.addEventListener('click', () => {
  if (isRecording) stopRecording();
  else startRecording();
});

// ---- Save note ----
saveBtn.addEventListener('click', async () => {
  if (!rawTranscript) return;
  const firstLine = rawTranscript.split('\n').find(l => l.trim()) || rawTranscript;
  const title = firstLine.slice(0, 60) || `Note ${Date.now()}`;
  const now = Date.now();
  const id = await dbSave({
    title,
    content: rawTranscript,
    language: detectedLanguage || null,
    createdAt: now,
    updatedAt: now
  });
  showDetail(id);
});

// ---- Views ----
function show(view) {
  recordView.hidden = view !== 'record';
  notesView.hidden = view !== 'notes';
  detailView.hidden = view !== 'detail';
  settingsView.hidden = view !== 'settings';
}

notesToggleBtn.addEventListener('click', async () => {
  show('notes');
  renderList(await dbAll());
});
backBtn.addEventListener('click', () => show('record'));
detailBackBtn.addEventListener('click', async () => {
  show('notes');
  renderList(await dbAll());
});

function renderList(notes) {
  const filter = searchInput.value.toLowerCase();
  const filtered = filter
    ? notes.filter(n => (n.title + n.content).toLowerCase().includes(filter))
    : notes;

  notesList.innerHTML = '';
  emptyState.hidden = filtered.length > 0;
  filtered.forEach(n => {
    const li = document.createElement('li');
    li.innerHTML = `
      <div class="note-title">${escape(n.title)}</div>
      <div class="note-preview">${escape(n.content.slice(0, 140))}</div>
      <div class="note-date">${new Date(n.updatedAt).toLocaleString()}</div>
    `;
    li.addEventListener('click', () => showDetail(n.id));
    notesList.appendChild(li);
  });
}

searchInput.addEventListener('input', async () => {
  renderList(await dbAll());
});

async function showDetail(id) {
  const note = await dbGet(id);
  if (!note) return;
  currentNoteId = id;
  detailTitle.value = note.title;
  detailContent.value = note.content;
  show('detail');
}

detailTitle.addEventListener('change', async () => {
  if (!currentNoteId) return;
  const note = await dbGet(currentNoteId);
  await dbSave({ ...note, title: detailTitle.value, updatedAt: Date.now() });
});
detailContent.addEventListener('change', async () => {
  if (!currentNoteId) return;
  const note = await dbGet(currentNoteId);
  await dbSave({ ...note, content: detailContent.value, updatedAt: Date.now() });
});

shareBtn.addEventListener('click', async () => {
  if (navigator.share) {
    await navigator.share({ title: detailTitle.value, text: detailContent.value });
  } else {
    await navigator.clipboard.writeText(detailContent.value);
    alert('Copied to clipboard');
  }
});

downloadBtn.addEventListener('click', async () => {
  if (!currentNoteId) return;
  const note = await dbGet(currentNoteId);
  if (!note) return;
  const md = `# ${note.title}\n\n*Created: ${new Date(note.createdAt).toISOString()}*\n${note.language ? `*Language: ${note.language}*\n` : ''}\n${note.content}\n`;
  const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
  const safeTitle = note.title.replace(/[^a-zA-Z0-9 _-]/g, '').trim().slice(0, 60) || `note-${note.id}`;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${safeTitle}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
});

deleteBtn.addEventListener('click', async () => {
  if (!currentNoteId) return;
  if (!confirm('Delete this note?')) return;
  await dbDelete(currentNoteId);
  currentNoteId = null;
  show('notes');
  renderList(await dbAll());
});

function escape(s) {
  return s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
}

// ---- Settings ----
settingsBtn.addEventListener('click', async () => {
  const notes = await dbAll();
  noteCountText.textContent = `${notes.length} ${notes.length === 1 ? 'note' : 'notes'}`;
  show('settings');
});
settingsBackBtn.addEventListener('click', () => show('record'));

licensesBtn.addEventListener('click', () => { licensesDialog.hidden = false; });
licensesCloseBtn.addEventListener('click', () => { licensesDialog.hidden = true; });

exportBtn.addEventListener('click', async () => {
  exportBtn.disabled = true;
  exportStatus.hidden = false;
  exportStatus.textContent = 'Building ZIP…';
  try {
    const notes = await dbAll();
    const zipBlob = await buildMarkdownZip(notes);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(zipBlob);
    a.download = `spext-notes-${Date.now()}.zip`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
    exportStatus.textContent = `Exported ${notes.length} notes.`;
  } catch (err) {
    exportStatus.textContent = `Export failed: ${err.message}`;
  } finally {
    exportBtn.disabled = false;
  }
});

// Build a ZIP using minimal manual ZIP format (no external lib needed for stored, no-compression files).
async function buildMarkdownZip(notes) {
  const files = notes.map(n => {
    const safeTitle = n.title.replace(/[^a-zA-Z0-9 _-]/g, '').trim() || `note-${n.id}`;
    const filename = `${safeTitle.slice(0, 120)}.md`;
    const md = `# ${n.title}\n\n*Created: ${new Date(n.createdAt).toISOString()}*\n\n${n.content}\n`;
    return { name: filename, data: new TextEncoder().encode(md) };
  });
  return await createStoredZip(files);
}

// Minimal ZIP encoder — stored (no compression) mode. Good for ~hundreds of small text files.
async function createStoredZip(files) {
  const enc = new TextEncoder();
  const parts = [];
  const central = [];
  let offset = 0;

  for (const f of files) {
    const nameBytes = enc.encode(f.name);
    const crc = crc32(f.data);
    const size = f.data.length;

    // Local file header
    const localHeader = new DataView(new ArrayBuffer(30));
    localHeader.setUint32(0, 0x04034b50, true);
    localHeader.setUint16(4, 20, true);
    localHeader.setUint16(6, 0, true);
    localHeader.setUint16(8, 0, true);
    localHeader.setUint16(10, 0, true);
    localHeader.setUint16(12, 0, true);
    localHeader.setUint32(14, crc, true);
    localHeader.setUint32(18, size, true);
    localHeader.setUint32(22, size, true);
    localHeader.setUint16(26, nameBytes.length, true);
    localHeader.setUint16(28, 0, true);
    parts.push(new Uint8Array(localHeader.buffer), nameBytes, f.data);

    // Central directory entry
    const cd = new DataView(new ArrayBuffer(46));
    cd.setUint32(0, 0x02014b50, true);
    cd.setUint16(4, 20, true);
    cd.setUint16(6, 20, true);
    cd.setUint16(8, 0, true);
    cd.setUint16(10, 0, true);
    cd.setUint16(12, 0, true);
    cd.setUint16(14, 0, true);
    cd.setUint32(16, crc, true);
    cd.setUint32(20, size, true);
    cd.setUint32(24, size, true);
    cd.setUint16(28, nameBytes.length, true);
    cd.setUint16(30, 0, true);
    cd.setUint16(32, 0, true);
    cd.setUint16(34, 0, true);
    cd.setUint16(36, 0, true);
    cd.setUint32(38, 0, true);
    cd.setUint32(42, offset, true);
    central.push(new Uint8Array(cd.buffer), nameBytes);

    offset += 30 + nameBytes.length + size;
  }

  const centralStart = offset;
  let centralSize = 0;
  for (const p of central) centralSize += p.length;

  const eocd = new DataView(new ArrayBuffer(22));
  eocd.setUint32(0, 0x06054b50, true);
  eocd.setUint16(4, 0, true);
  eocd.setUint16(6, 0, true);
  eocd.setUint16(8, files.length, true);
  eocd.setUint16(10, files.length, true);
  eocd.setUint32(12, centralSize, true);
  eocd.setUint32(16, centralStart, true);
  eocd.setUint16(20, 0, true);

  return new Blob([...parts, ...central, new Uint8Array(eocd.buffer)], { type: 'application/zip' });
}

// CRC32 for ZIP — table-based.
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();
function crc32(data) {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// ---- PWA install prompt ----
let deferredInstallPrompt = null;
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredInstallPrompt = e;
  installBtn.disabled = false;
  installBtn.textContent = 'Install Spext';
});
installBtn.addEventListener('click', async () => {
  if (!deferredInstallPrompt) return;
  deferredInstallPrompt.prompt();
  const result = await deferredInstallPrompt.userChoice;
  deferredInstallPrompt = null;
  installBtn.disabled = true;
  installBtn.textContent = result.outcome === 'accepted' ? 'Installed' : 'Use browser menu';
});

// ---- Service Worker (offline cache) ----
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch(err => {
    console.warn('SW registration failed', err);
  });
}

// ---- Bootstrap ----
loadWhisper();

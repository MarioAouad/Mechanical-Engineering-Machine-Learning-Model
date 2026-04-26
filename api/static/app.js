// ===================================================================
// Acoustic Anomaly Detection — Frontend Logic
// ===================================================================

const MACHINE_CONFIG = [
    { id: 'ToyCar',   name: 'Toy Car',   icon: '🚗', color: '#6366f1' },
    { id: 'ToyTrain', name: 'Toy Train', icon: '🚂', color: '#8b5cf6' },
    { id: 'bearing',  name: 'Bearing',   icon: '⚙️', color: '#06b6d4' },
    { id: 'fan',      name: 'Fan',       icon: '🌀', color: '#10b981' },
    { id: 'gearbox',  name: 'Gearbox',   icon: '🔩', color: '#f59e0b' },
    { id: 'slider',   name: 'Slider',    icon: '📏', color: '#ec4899' },
    { id: 'valve',    name: 'Valve',     icon: '🔧', color: '#ef4444' },
];

let selectedMachine = null;
let selectedFile = null;
let machineMetadata = {};

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
    createParticles();
    fetchMachines();
    buildMachineButtons();
    bindEvents();
});

// --- Background Particles ---
function createParticles() {
    const container = document.getElementById('bgParticles');
    const colors = ['rgba(99,102,241,0.15)', 'rgba(6,182,212,0.12)', 'rgba(139,92,246,0.1)'];
    for (let i = 0; i < 18; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.width = p.style.height = Math.random() * 6 + 2 + 'px';
        p.style.left = Math.random() * 100 + '%';
        p.style.top = Math.random() * 100 + '%';
        p.style.background = colors[i % colors.length];
        p.style.animationDelay = Math.random() * 12 + 's';
        p.style.animationDuration = 10 + Math.random() * 8 + 's';
        container.appendChild(p);
    }
}

// --- Fetch machine metadata from API ---
async function fetchMachines() {
    try {
        const res = await fetch('/machines');
        if (res.ok) {
            const data = await res.json();
            machineMetadata = data.machines || {};
            // Update pipeline labels
            MACHINE_CONFIG.forEach(m => {
                const meta = machineMetadata[m.id];
                if (meta) {
                    const btn = document.getElementById('btn-' + m.id);
                    if (btn) {
                        const pipelineEl = btn.querySelector('.pipeline');
                        if (pipelineEl) pipelineEl.textContent = `${meta.pipeline} · ${meta.strategy}`;
                    }
                }
            });
        }
    } catch (e) {
        console.warn('Could not fetch machine metadata:', e);
    }
}

// --- Build Buttons ---
function buildMachineButtons() {
    const grid = document.getElementById('machinesGrid');
    MACHINE_CONFIG.forEach(m => {
        const btn = document.createElement('button');
        btn.className = 'machine-btn';
        btn.id = 'btn-' + m.id;
        btn.innerHTML = `
            <div class="machine-icon" style="background:${m.color}15; color:${m.color}">${m.icon}</div>
            <div class="machine-label">
                <span class="name">${m.name}</span>
                <span class="pipeline">Loading…</span>
            </div>
        `;
        btn.addEventListener('click', () => selectMachine(m));
        grid.appendChild(btn);
    });
}

// --- Select Machine ---
function selectMachine(machine) {
    selectedMachine = machine;
    selectedFile = null;

    // Highlight
    document.querySelectorAll('.machine-btn').forEach(b => b.classList.remove('selected'));
    document.getElementById('btn-' + machine.id).classList.add('selected');

    // Show upload step
    document.getElementById('selectedMachineName').textContent = machine.name;
    show('stepUpload');
    hide('stepResult');

    // Reset upload state
    document.getElementById('audioInput').value = '';
    document.getElementById('fileInfo').classList.add('hidden');
    document.getElementById('btnPredict').classList.add('hidden');
    document.getElementById('uploadZone').style.display = '';
}

// --- Events ---
function bindEvents() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('audioInput');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => { if (input.files.length) handleFile(input.files[0]); });

    document.getElementById('btnClear').addEventListener('click', clearFile);
    document.getElementById('btnPredict').addEventListener('click', runPrediction);
    document.getElementById('btnReset').addEventListener('click', resetAll);
}

function handleFile(file) {
    if (!file.name.toLowerCase().endsWith('.wav')) {
        alert('Please upload a .wav audio file.');
        return;
    }
    selectedFile = file;
    document.getElementById('fileName').textContent = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
    document.getElementById('fileInfo').classList.remove('hidden');
    document.getElementById('btnPredict').classList.remove('hidden');
    document.getElementById('uploadZone').style.display = 'none';
}

function clearFile() {
    selectedFile = null;
    document.getElementById('audioInput').value = '';
    document.getElementById('fileInfo').classList.add('hidden');
    document.getElementById('btnPredict').classList.add('hidden');
    document.getElementById('uploadZone').style.display = '';
}

// --- Prediction ---
async function runPrediction() {
    if (!selectedMachine || !selectedFile) return;

    const btn = document.getElementById('btnPredict');
    const loader = document.getElementById('predictLoader');
    const textEl = btn.querySelector('.btn-predict-text');

    btn.disabled = true;
    textEl.textContent = 'Analyzing…';
    loader.classList.remove('hidden');

    try {
        const form = new FormData();
        form.append('audio', selectedFile);

        const res = await fetch(`/predict?machine_type=${encodeURIComponent(selectedMachine.id)}`, {
            method: 'POST',
            body: form,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Prediction failed');
        }

        const result = await res.json();
        showResult(result);

    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btn.disabled = false;
        textEl.textContent = 'Analyze Audio';
        loader.classList.add('hidden');
    }
}

// --- Display Result ---
function showResult(r) {
    show('stepResult');

    const badge = document.getElementById('resultBadge');
    const isAnomaly = r.is_anomaly;

    badge.className = 'result-badge ' + (isAnomaly ? 'anomaly' : 'normal');
    badge.innerHTML = isAnomaly
        ? '⚠️ ANOMALY DETECTED'
        : '✅ NORMAL';

    const details = document.getElementById('resultDetails');
    details.innerHTML = `
        <div class="detail-item">
            <div class="detail-label">Anomaly Score</div>
            <div class="detail-value">${r.anomaly_score.toFixed(6)}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Threshold</div>
            <div class="detail-value">${r.threshold.toFixed(6)}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Confidence</div>
            <div class="detail-value">${r.confidence.toFixed(2)}σ</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Strategy</div>
            <div class="detail-value">${r.strategy}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Pipeline</div>
            <div class="detail-value">${r.pipeline}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Latency</div>
            <div class="detail-value">${r.latency_ms} ms</div>
        </div>
    `;

    // Scroll to result
    document.getElementById('stepResult').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// --- Reset ---
function resetAll() {
    selectedMachine = null;
    selectedFile = null;
    document.querySelectorAll('.machine-btn').forEach(b => b.classList.remove('selected'));
    hide('stepUpload');
    hide('stepResult');
    clearFile();
}

// --- Helpers ---
function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }

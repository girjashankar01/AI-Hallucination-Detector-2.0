const API = 'https://ai-hallucination-detector-production.up.railway.app';
const $ = id => document.getElementById(id);

const textarea = $('inputText');
const charEl   = $('charCount');
const btn      = $('analyzeBtn');
const loadEl   = $('loading');
const resEl    = $('results');
const toastEl  = $('errorToast');
const hintEl   = $('inputHint');

let lastData = null; // store for copy

// ── paste from clipboard ──────────────────────────────────
$('pasteBtn').addEventListener('click', async () => {
    try {
        const text = await navigator.clipboard.readText();
        if (text) {
            textarea.value = text.slice(0, 500);
            textarea.dispatchEvent(new Event('input'));
            textarea.focus();
        }
    } catch {
        showErr('Clipboard access denied — paste manually with Ctrl/Cmd+V');
    }
});

// ── copy results to clipboard ─────────────────────────────
$('copyBtn').addEventListener('click', (e) => {
    e.stopPropagation();
    if (!lastData) return;
    const d = lastData;
    const lines = [
        `Hallucination Detector — Results`,
        `Overall: ${d.overall_score} (${d.overall_label})`,
        `Topic: ${d.topic}`,
        `${d.hallucinated_count} hallucinated, ${d.grounded_count} grounded, ${d.sentence_count} sentences`,
        '',
        ...d.results.map((r, i) =>
            `[${i+1}] ${r.sentence}\n    Score: ${r.final_score} | Label: ${r.label} | NLI: ${r.nli_verdict}\n    Grounding: ${r.grounding_score} | Consistency: ${r.consistency_score} | Embedding: ${r.embedding_score} | NLI: ${r.nli_score}`
        ),
    ];
    navigator.clipboard.writeText(lines.join('\n')).then(() => {
        const label = $('copyLabel');
        const copyBtn = $('copyBtn');
        label.textContent = 'Copied!';
        copyBtn.classList.add('copied');
        setTimeout(() => { label.textContent = 'Copy'; copyBtn.classList.remove('copied'); }, 1800);
    });
});

// ── Enter key to analyze ──────────────────────────────────
textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!btn.disabled) analyze();
    }
});

// ── char counter + validation ─────────────────────────────
textarea.addEventListener('input', () => {
    const n = textarea.value.length;
    charEl.textContent = `${n} / 500`;
    charEl.className = 'char-count' + (n > 450 ? ' over' : n > 350 ? ' warn' : '');
    validateInput(textarea.value);
});

// ── loading steps ─────────────────────────────────────────
const STEPS = [
    'Inferring topic from text…',
    'Fetching Wikipedia facts…',
    'Building vector store…',
    'Running consistency checks…',
    'Running NLI classification…',
    'Computing final scores…',
];
let stepI = 0, stepT;

function stepsOn() {
    stepI = 0;
    const el = $('loadingStep');
    el.textContent = STEPS[0];
    stepT = setInterval(() => {
        stepI = (stepI + 1) % STEPS.length;
        el.style.opacity = '0';
        setTimeout(() => { el.textContent = STEPS[stepI]; el.style.opacity = '1'; }, 150);
    }, 3000);
}

function stepsOff() { clearInterval(stepT); }

// ── input validation ──────────────────────────────────────
function validateInput(text) {
    const trimmed = text.trim();
    if (!trimmed) {
        setHint('', '');
        return true;
    }

    const wordCount = trimmed.split(/\s+/).length;

    if (wordCount < 3) {
        setHint('Too short — enter at least a full sentence', 'err');
        return false;
    }

    if (wordCount < 6) {
        setHint('Very short input — results may be unreliable', 'warn');
        return true; // allow but warn
    }

    // Check if it looks like a question or command, not a factual claim
    const lower = trimmed.toLowerCase();
    if (/^(who|what|when|where|why|how|is|are|can|do|does|did|will|should|would|could)\s/i.test(lower) && trimmed.endsWith('?')) {
        setHint('This looks like a question — enter factual claims instead', 'warn');
        return true;
    }

    if (/^(please|hey|hi|hello|help|tell me|explain|describe|write|generate|create|make)\s/i.test(lower)) {
        setHint('This looks like a prompt — enter factual statements to verify', 'warn');
        return true;
    }

    setHint('', '');
    return true;
}

function setHint(msg, cls) {
    hintEl.textContent = msg;
    hintEl.className = 'input-hint' + (cls ? ` ${cls}` : '');
}

// ── analyze ───────────────────────────────────────────────
async function analyze() {
    const text = textarea.value.trim();
    if (!text) return showErr('Enter some text to analyze.');

    const wordCount = text.split(/\s+/).length;
    if (wordCount < 3) {
        setHint('Too short — enter at least a full sentence', 'err');
        return;
    }

    btn.disabled = true;
    loadEl.classList.add('show');
    resEl.classList.remove('show');
    toastEl.classList.remove('show');
    stepsOn();

    try {
        const res=await fetch(`${API}/analyze`,{
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
    });
        const data = await res.json();
        if (!res.ok) return showErr(data.detail || 'Analysis failed.');
        lastData = data;
        render(data);
    } catch {
        showErr('Cannot reach API — is the server running?');
    } finally {
        btn.disabled = false;
        loadEl.classList.remove('show');
        stepsOff();
    }
}

// ── color helpers ─────────────────────────────────────────
function mc(v) {
    if (v >= 0.7) return 'var(--green)';
    if (v >= 0.5) return 'var(--amber)';
    return 'var(--red)';
}

function mcRaw(v) {
    if (v >= 0.7) return '#00ff88';
    if (v >= 0.5) return '#f0a030';
    return '#ff4060';
}

// ── SVG gauge ─────────────────────────────────────────────
function gauge(score, hall) {
    const r = 50, c = 2 * Math.PI * r;
    const off = c * (1 - score);
    const color = hall ? '#ff4060' : '#00ff88';

    return `
        <svg viewBox="0 0 120 120" width="120" height="120">
            <circle class="g-track" cx="60" cy="60" r="${r}" />
            <circle class="g-fill" cx="60" cy="60" r="${r}"
                stroke="${color}"
                stroke-dasharray="${c}"
                stroke-dashoffset="${off}" />
        </svg>
        <span class="gauge-number" style="color:${color}">${score.toFixed(2)}</span>
        <span class="gauge-label">Score</span>`;
}

// ── render ────────────────────────────────────────────────
function render(data) {
    const hall  = data.overall_label === 'likely hallucinated';
    const score = data.overall_score;
    const cls   = hall ? 'hall' : 'grounded';
    const color = hall ? 'var(--red)' : 'var(--green)';
    const rawC  = hall ? '#ff4060' : '#00ff88';
    const pct   = Math.round(score * 100);

    // gauge
    $('gaugeWrap').innerHTML = gauge(score, hall);

    // verdict label
    const vl = $('verdictLabel');
    vl.className = `verdict-label ${cls}`;
    vl.textContent = hall ? '⚠ Likely Hallucinated' : '✓ Mostly Grounded';

    // meta tags
    $('verdictMeta').innerHTML = `
        <span class="v-tag">${data.topic}</span>
        <span class="v-tag"><span class="v-dot r"></span>${data.hallucinated_count} hallucinated</span>
        <span class="v-tag"><span class="v-dot g"></span>${data.grounded_count} grounded</span>
        <span class="v-tag">${data.sentence_count} sentence${data.sentence_count !== 1 ? 's' : ''}</span>`;

    // overall horizontal bar
    const barFill = $('overallBarFill');
    barFill.style.width = pct + '%';
    barFill.style.background = rawC;
    $('overallBarPct').textContent = pct + '%';
    $('overallBarPct').style.color = color;

    // metric breakdown
    const r2 = data.results;
    const avg = k => r2.length ? r2.reduce((s, x) => s + x[k], 0) / r2.length : 0;
    const metrics = [
        ['Grounding',   avg('grounding_score')],
        ['Consistency', avg('consistency_score')],
        ['Embedding',   avg('embedding_score')],
        ['NLI',         avg('nli_score')],
    ];

    $('metricRow').innerHTML = metrics.map(([name, val]) => `
        <div class="m-card">
            <div class="m-card-top">
                <span class="m-name">${name}</span>
                <span class="m-val" style="color:${mc(val)}">${val.toFixed(2)}</span>
            </div>
            <div class="m-bar">
                <div class="m-bar-fill" style="width:${Math.round(val*100)}%;background:${mc(val)}"></div>
            </div>
        </div>`).join('');

    // sentence count
    $('sentenceCount').textContent = `${data.sentence_count} sentences`;

    // sentence cards
    const list = $('sentenceList');
    list.innerHTML = '';

    data.results.forEach((item, i) => {
        const ih = item.label === 'hallucinated';
        const ic = ih ? 'hall' : 'grounded';

        const nv = item.nli_verdict || 'NEUTRAL';
        let nc = 'neutral';
        if (nv === 'CONTRADICTION') nc = 'hall';
        else if (nv === 'ENTAILMENT') nc = 'grounded';

        const contraH = (item.contradicting_fact && nv === 'CONTRADICTION')
            ? `<div class="nli-contra">${item.contradicting_fact}</div>` : '';

        const el = document.createElement('div');
        el.className = 'sc';
        el.innerHTML = `
            <div class="sc-top">
                <span class="sc-dot ${ic}"></span>
                <span class="sc-txt">${item.sentence}</span>
                <span class="sc-score ${ic}">${item.final_score}</span>
                <span class="sc-chev" id="ch-${i}">▾</span>
            </div>
            <div class="sc-detail" id="dt-${i}">
                <div class="sc-metrics">
                    ${mCell('Grounding', item.grounding_score)}
                    ${mCell('Consistency', item.consistency_score)}
                    ${mCell('Embedding', item.embedding_score)}
                    ${mCell('NLI', item.nli_score)}
                </div>
                <div class="sc-block">
                    <div class="sc-block-title">Closest Wikipedia Fact</div>
                    <div class="sc-block-body">${item.evidence}</div>
                </div>
                <div class="sc-block">
                    <div class="sc-block-title">Gemini Responses (×3)</div>
                    ${item.consistency_responses.map(r => `<div class="sc-resp">${r}</div>`).join('')}
                </div>
                <div class="sc-block">
                    <div class="sc-block-title">NLI Classification</div>
                    <div class="nli-row">
                        <span class="nli-badge ${nc}">${nv}</span>
                        <span class="nli-sv" style="color:${mc(item.nli_score||0.5)}">score: ${item.nli_score}</span>
                    </div>
                    ${contraH}
                </div>
            </div>`;

        el.addEventListener('click', () => {
            $(`dt-${i}`).classList.toggle('open');
            $(`ch-${i}`).classList.toggle('open');
        });

        list.appendChild(el);
    });

    resEl.classList.add('show');
}

function mCell(name, val) {
    const c = mc(val), p = Math.round(val * 100);
    return `<div class="sc-m">
        <div class="sc-m-val" style="color:${c}">${val}</div>
        <div class="sc-m-name">${name}</div>
        <div class="sc-m-bar"><div class="sc-m-bar-fill" style="width:${p}%;background:${c}"></div></div>
    </div>`;
}

function showErr(msg) {
    $('errorMsg').textContent = msg;
    toastEl.classList.add('show');
    setTimeout(() => toastEl.classList.remove('show'), 5000);
}

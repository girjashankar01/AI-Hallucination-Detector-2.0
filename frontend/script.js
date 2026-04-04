// Auto-detect local dev vs production
const API = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : 'https://ai-hallucination-detector-production.up.railway.app';

const $ = id => document.getElementById(id);

const textarea = $('inputText');
const charEl   = $('charCount');
const btn      = $('analyzeBtn');
const loadEl   = $('loading');
const resEl    = $('results');
const toastEl  = $('errorToast');
const hintEl   = $('inputHint');

let lastData = null;

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
        `Domain: ${d.domain} | Topic: ${d.topic}`,
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

// ── loading steps — domain-aware ──────────────────────────
let _loadDomain = 'general';

function getSteps(domain) {
    const src = { medical: 'PubMed abstracts', legal: 'CourtListener opinions',
                  financial: 'Wikipedia', general: 'Wikipedia' }[domain] || 'Wikipedia';
    return [
        'Inferring topic from text...',
        `Fetching ${src}...`,
        'Building vector store...',
        'Running consistency checks...',
        'Running NLI classification...',
        'Computing final scores...',
    ];
}

let stepI = 0, stepT;

function stepsOn() {
    _loadDomain = 'general';
    stepI = 0;
    const el = $('loadingStep');
    el.textContent = getSteps('general')[0];
    stepT = setInterval(() => {
        stepI = (stepI + 1) % 6;
        el.style.opacity = '0';
        setTimeout(() => {
            el.textContent = getSteps(_loadDomain)[stepI];
            el.style.opacity = '1';
        }, 150);
    }, 3000);
}

function stepsOff() { clearInterval(stepT); }

// ── input validation ──────────────────────────────────────
function validateInput(text) {
    const trimmed = text.trim();
    if (!trimmed) { setHint('', ''); return true; }
    const wordCount = trimmed.split(/\s+/).length;
    if (wordCount < 3) { setHint('Too short — enter at least a full sentence', 'err'); return false; }
    if (wordCount < 6) { setHint('Very short input — results may be unreliable', 'warn'); return true; }
    const lower = trimmed.toLowerCase();
    if (/^(who|what|when|where|why|how|is|are|can|do|does|did|will|should|would|could)\s/i.test(lower) && trimmed.endsWith('?')) {
        setHint('This looks like a question — enter factual claims instead', 'warn'); return true;
    }
    if (/^(please|hey|hi|hello|help|tell me|explain|describe|write|generate|create|make)\s/i.test(lower)) {
        setHint('This looks like a prompt — enter factual statements to verify', 'warn'); return true;
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
    if (text.split(/\s+/).length < 3) { setHint('Too short — enter at least a full sentence', 'err'); return; }

    btn.disabled = true;
    loadEl.classList.add('show');
    resEl.classList.remove('show');
    toastEl.classList.remove('show');
    stepsOn();

    try {
        const res = await fetch(`${API}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        const data = await res.json();
        if (!res.ok) return showErr(data.detail || 'Analysis failed.');
        _loadDomain = data.domain || 'general';
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
    if (v >= 0.65) return 'var(--green)';
    if (v >= 0.40) return 'var(--amber)';
    return 'var(--red)';
}
function mcRaw(v) {
    if (v >= 0.65) return '#00ff88';
    if (v >= 0.40) return '#f0a030';
    return '#ff4060';
}
// NLI scores are discrete: 0.0=contradiction, 0.5=neutral, 1.0=entailment
function mcNli(v) {
    if (v >= 0.9) return 'var(--green)';
    if (v >= 0.4) return 'var(--amber)';
    return 'var(--red)';
}
function domainColor(domain) {
    return { medical: '#5dcaa5', legal: '#afa9ec', financial: '#f0a030', general: '#00ff88' }[domain] || '#00ff88';
}

// ── SVG gauge — supports mixed (amber) verdict ────────────
function gauge(score, label) {
    const r = 50, c = 2 * Math.PI * r;
    const off = c * (1 - score);
    const isHall  = label === 'likely hallucinated';
    const isMixed = label && label.startsWith('mixed');
    const color = isHall ? '#ff4060' : isMixed ? '#f0a030' : '#00ff88';
    return `
        <svg viewBox="0 0 120 120" width="120" height="120">
            <circle class="g-track" cx="60" cy="60" r="${r}" />
            <circle class="g-fill" cx="60" cy="60" r="${r}"
                stroke="${color}" stroke-dasharray="${c}" stroke-dashoffset="${off}" />
        </svg>
        <span class="gauge-number" style="color:${color}">${score.toFixed(2)}</span>
        <span class="gauge-label">Score</span>`;
}

// ── render ────────────────────────────────────────────────
function render(data) {
    const label   = data.overall_label || '';
    const isHall  = label === 'likely hallucinated';
    const isMixed = label.startsWith('mixed');
    const score   = data.overall_score;
    const pct     = Math.round(score * 100);

    const cls   = isHall ? 'hall' : isMixed ? 'mixed' : 'grounded';
    const rawC  = isHall ? '#ff4060' : isMixed ? '#f0a030' : '#00ff88';
    const color = isHall ? 'var(--red)' : isMixed ? 'var(--amber)' : 'var(--green)';

    const verdictText = isHall  ? 'Likely Hallucinated'
                      : isMixed ? label.charAt(0).toUpperCase() + label.slice(1)
                      : 'Mostly Grounded';
    const verdictIcon = isHall ? '' : isMixed ? '' : '';

    $('gaugeWrap').innerHTML = gauge(score, label);

    const vl = $('verdictLabel');
    vl.className = `verdict-label ${cls}`;
    vl.textContent = `${verdictIcon} ${verdictText}`;

    $('verdictMeta').innerHTML = `
        <span class="v-tag domain-badge domain-${data.domain}">${data.domain}</span>
        <span class="v-tag">${data.topic}</span>
        <span class="v-tag"><span class="v-dot r"></span>${data.hallucinated_count} hallucinated</span>
        <span class="v-tag"><span class="v-dot g"></span>${data.grounded_count} grounded</span>
        <span class="v-tag">${data.sentence_count} sentence${data.sentence_count !== 1 ? 's' : ''}</span>`;

    const barFill = $('overallBarFill');
    barFill.style.width = pct + '%';
    barFill.style.background = rawC;
    $('overallBarPct').textContent = pct + '%';
    $('overallBarPct').style.color = color;

    // Metric cards — NLI first (highest weight), correct color function per metric
    const r2  = data.results;
    const avg = k => r2.length ? r2.reduce((s, x) => s + (x[k] || 0), 0) / r2.length : 0;
    const metrics = [
        { name: 'NLI',         val: avg('nli_score'),         colorFn: mcNli, weight: '40%' },
        { name: 'Consistency', val: avg('consistency_score'), colorFn: mc,    weight: '35%' },
        { name: 'Grounding',   val: avg('grounding_score'),   colorFn: mc,    weight: '15%' },
        { name: 'Embedding',   val: avg('embedding_score'),   colorFn: mc,    weight: '10%' },
    ];

    $('metricRow').innerHTML = metrics.map(({ name, val, colorFn, weight }) => `
        <div class="m-card">
            <div class="m-card-top">
                <span class="m-name">${name} <span class="m-weight">${weight}</span></span>
                <span class="m-val" style="color:${colorFn(val)}">${val.toFixed(2)}</span>
            </div>
            <div class="m-bar">
                <div class="m-bar-fill" style="width:${Math.round(val*100)}%;background:${colorFn(val)}"></div>
            </div>
        </div>`).join('');

    $('sentenceCount').textContent = `${data.sentence_count} sentences`;

    const list = $('sentenceList');
    list.innerHTML = '';
    const srcColor = domainColor(data.domain);

    data.results.forEach((item, i) => {
        const ih = item.label === 'hallucinated';
        const ic = ih ? 'hall' : 'grounded';
        const nv = item.nli_verdict || 'NEUTRAL';
        let nc = 'neutral';
        if (nv === 'CONTRADICTION') nc = 'hall';
        else if (nv === 'ENTAILMENT') nc = 'grounded';

        const contraH = (item.contradicting_fact && nv === 'CONTRADICTION')
            ? `<div class="nli-contra">${item.contradicting_fact}</div>` : '';

        // Filter raw BART-MNLI debug lines from consistency_responses:
        // "NLI (hypothesis_template): P(supported)=0.056, top='refuted'" is
        // internal backend output that leaked into the field — hide it.
        const cleanResponses = (item.consistency_responses || []).filter(r => {
            if (!r) return false;
            if (r.includes('hypothesis_template') || r.includes('P(supported)')) return false;
            if (r.trim().length < 10) return false;
            return true;
        });

        const el = document.createElement('div');
        el.className = 'sc';
        el.innerHTML = `
            <div class="sc-top">
                <span class="sc-dot ${ic}"></span>
                <span class="sc-txt">${item.sentence}</span>
                <span class="sc-score ${ic}">${item.final_score}</span>
                <span class="sc-chev" id="ch-${i}">&#9662;</span>
            </div>
            <div class="sc-detail" id="dt-${i}">
                <div class="sc-metrics">
                    ${mCell('NLI',         item.nli_score,         mcNli)}
                    ${mCell('Consistency', item.consistency_score, mc)}
                    ${mCell('Grounding',   item.grounding_score,   mc)}
                    ${mCell('Embedding',   item.embedding_score,   mc)}
                </div>
                <div class="sc-block">
                    <div class="sc-block-title">Closest Source Fact</div>
                    <div class="sc-block-body">${item.evidence}</div>
                    ${item.evidence_url
                        ? `<a href="${item.evidence_url}" target="_blank" class="src-link" style="color:${srcColor}">${item.evidence_source || 'Source'} &#8599;</a>`
                        : (item.evidence_source ? `<span class="src-label">${item.evidence_source}</span>` : '')
                    }
                </div>
                ${cleanResponses.length ? `
                <div class="sc-block">
                    <div class="sc-block-title">Consistency Responses</div>
                    ${cleanResponses.map(r => `<div class="sc-resp">${r}</div>`).join('')}
                </div>` : ''}
                <div class="sc-block">
                    <div class="sc-block-title">NLI Classification</div>
                    <div class="nli-row">
                        <span class="nli-badge ${nc}">${nv}</span>
                        <span class="nli-sv" style="color:${mcNli(item.nli_score || 0.5)}">score: ${item.nli_score}</span>
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

function mCell(name, val, colorFn) {
    const c = colorFn(val), p = Math.round(val * 100);
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
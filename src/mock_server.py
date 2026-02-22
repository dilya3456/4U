import json
import re
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Helpers
# =========================
def normalize_text(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    s = re.sub(r"\s+", " ", s).strip()
    return s


ICD_EXACT_RE = re.compile(r"\b([A-ZА-Я][0-9]{2}(?:\.[0-9A-ZА-Я]{1,2})?)\b")


def is_range_code(code: str) -> bool:
    # e.g., O00-O99
    return bool(re.fullmatch(r"[A-ZА-Я][0-9]{2}\s*-\s*[A-ZА-Я][0-9]{2}", (code or "").strip().upper()))


def is_general_code(code: str) -> bool:
    # J18 is more general than J18.9
    return bool(re.fullmatch(r"[A-ZА-Я][0-9]{2}", (code or "").strip().upper()))


def extract_exact_icds(text: str) -> list[str]:
    if not text:
        return []
    t = text.upper().replace("Ё", "Е")
    found = ICD_EXACT_RE.findall(t)
    out = []
    for x in found:
        if x not in out:
            out.append(x)
    return out


# =========================
# Data loading
# =========================
DATA_DIR = Path("corpus")

protocol_texts: list[str] = []
protocol_meta: list[dict] = []


def add_obj(obj: dict):
    if not isinstance(obj, dict):
        return
    if "text" not in obj or "icd_codes" not in obj:
        return

    title = obj.get("title", "") or ""
    text = obj.get("text", "") or ""

    full_text = f"{title}\n{text}".strip()
    protocol_texts.append(normalize_text(full_text))

    icds_from_meta = obj.get("icd_codes", []) or []
    icds_from_text = extract_exact_icds(text)

    merged = []
    for icd in list(icds_from_meta) + list(icds_from_text):
        if not icd:
            continue
        icd_norm = icd.strip().upper()
        if icd_norm and icd_norm not in merged:
            merged.append(icd_norm)

    obj["_icd_merged"] = merged
    protocol_meta.append(obj)


def load_corpus():
    protocol_texts.clear()
    protocol_meta.clear()

    for file in DATA_DIR.rglob("*"):
        if not file.is_file():
            continue
        suffix = file.suffix.lower()
        try:
            if suffix == ".jsonl":
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        add_obj(json.loads(line))
            elif suffix == ".json":
                with open(file, "r", encoding="utf-8") as f:
                    add_obj(json.load(f))
        except Exception:
            continue


load_corpus()
print(f"Loaded protocols: {len(protocol_texts)}")


# =========================
# Vectorizers (hybrid)
# =========================
vectorizer_word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    token_pattern=r"(?u)\b[а-яa-z0-9\-]{2,}\b",
)
vectorizer_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
)

tfidf_word = vectorizer_word.fit_transform(protocol_texts) if protocol_texts else None
tfidf_char = vectorizer_char.fit_transform(protocol_texts) if protocol_texts else None


def topk_indices(query_vec, matrix, topk: int) -> list[int]:
    sims = cosine_similarity(query_vec, matrix).flatten()
    if topk >= len(sims):
        return sims.argsort()[::-1].tolist()
    return sims.argsort()[-topk:][::-1].tolist()


def compute_ranks(n_docs: int, ranked_indices: list[int]) -> list[int]:
    ranks = [0] * n_docs
    for rank, idx in enumerate(ranked_indices, start=1):
        ranks[idx] = rank
    return ranks


def score_icds(top_doc_indices: list[int], doc_scores: dict[int, float]) -> list[tuple[str, float, int]]:
    icd_score: dict[str, float] = {}
    icd_best_doc: dict[str, int] = {}

    for idx in top_doc_indices:
        base = float(doc_scores.get(idx, 0.0))
        icds = protocol_meta[idx].get("_icd_merged", []) or []
        for icd in icds:
            s = base
            if is_range_code(icd):
                s *= 0.25
            elif is_general_code(icd):
                s *= 0.75
            else:
                s *= 1.15

            icd_score[icd] = icd_score.get(icd, 0.0) + s
            if icd not in icd_best_doc or base > doc_scores.get(icd_best_doc[icd], -1.0):
                icd_best_doc[icd] = idx

    ranked = sorted(icd_score.items(), key=lambda x: x[1], reverse=True)
    out = []
    for icd, sc in ranked:
        out.append((icd, float(sc), icd_best_doc.get(icd, top_doc_indices[0] if top_doc_indices else 0)))
    return out


def make_explanation(icd: str, doc_idx: int) -> str:
    obj = protocol_meta[doc_idx]
    pid = obj.get("protocol_id", "unknown")
    title = obj.get("title", "") or ""
    text = (obj.get("text", "") or "").strip()

    t_up = text.upper().replace("Ё", "Е")
    pos = t_up.find(icd.upper())
    if pos != -1:
        start = max(0, pos - 140)
        end = min(len(text), pos + 140)
        snippet = text[start:end].replace("\n", " ")
    else:
        snippet = text[:280].replace("\n", " ")

    snippet = re.sub(r"\s+", " ", snippet).strip()
    return f"Protocol: {pid}. {title}. Evidence: “{snippet}”"


# =========================
# FastAPI
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🏥 Diagnostic Server (HYBRID TF-IDF + UI)")
    print("=" * 60)
    print("UI:   /")
    print("API:  POST /diagnose")
    print("Docs: /docs")
    print("=" * 60)
    yield


app = FastAPI(title="Diagnostic Server", lifespan=lifespan)


class DiagnoseRequest(BaseModel):
    symptoms: Optional[str] = ""


class Diagnosis(BaseModel):
    rank: int
    diagnosis: str
    icd10_code: str
    explanation: str


class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]


@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(
        """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <script src="https://cdn.tailwindcss.com"></script>
  <title>MedAI</title>
</head>

<body class="min-h-screen bg-slate-950 text-white overflow-x-hidden">
  <!-- NAVBAR -->
  <nav class="w-full px-8 py-5 flex justify-between items-center border-b border-slate-800 backdrop-blur-xl bg-slate-950/70 sticky top-0 z-50">
    <div class="text-xl font-semibold tracking-tight">MedAI</div>
    <div class="text-sm text-slate-400">Clinical Intelligence Platform</div>
  </nav>

  <!-- HERO -->
  <section class="relative text-center py-24 px-6">
    <div class="absolute inset-0 -z-10">
      <div class="absolute top-0 left-1/2 -translate-x-1/2 w-[700px] h-[400px] bg-blue-600 opacity-20 blur-[120px] rounded-full"></div>
    </div>

    <div class="inline-block mb-6 px-4 py-1 rounded-full bg-slate-800 text-blue-400 text-sm">
      AI Powered Clinical System
    </div>

    <h1 class="text-6xl font-bold tracking-tight leading-tight">
      Интеллектуальная система
      <span class="block bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
        предварительной диагностики
      </span>
    </h1>

    <p class="mt-6 text-lg text-slate-400 max-w-2xl mx-auto">
      Профессиональный анализ симптомов с формированием структурированного диагностического отчёта на основе клинических протоколов РК.
    </p>
  </section>

  <!-- APP CARD -->
  <section class="max-w-4xl mx-auto px-6 pb-24">
    <div class="bg-slate-900 border border-slate-800 rounded-3xl p-10 shadow-2xl backdrop-blur-xl">
      <label class="block text-lg font-medium mb-4 text-slate-300">
        Введите симптомы пациента
      </label>

      <textarea id="symptoms"
        class="w-full p-6 rounded-2xl bg-slate-950 border border-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 transition text-lg resize-none text-white"
        rows="5"
        placeholder="Например: боль в груди, высокая температура, слабость, одышка..."></textarea>

      <button id="btn"
        class="mt-6 w-full py-4 rounded-2xl text-lg font-semibold bg-gradient-to-r from-blue-600 to-cyan-500 hover:opacity-90 transition-all duration-300">
        Запустить диагностику
      </button>

      <div id="bar" class="hidden mt-6 h-2 w-full bg-slate-800 rounded-full overflow-hidden">
        <div class="h-full bg-gradient-to-r from-blue-500 to-cyan-400 animate-pulse w-3/4"></div>
      </div>
    </div>

    <div id="result" class="mt-16 space-y-8"></div>
  </section>

  <footer class="border-t border-slate-800 py-6 text-center text-sm text-slate-500">
    © 2026 MedAI Systems — AI Clinical Intelligence
  </footer>

<script>
  const btn = document.getElementById("btn");
  const ta = document.getElementById("symptoms");
  const res = document.getElementById("result");
  const bar = document.getElementById("bar");

  function escapeHtml(s) {
    return (s || "").replace(/[&<>"']/g, (c) => ({
      "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"
    }[c]));
  }

  btn.addEventListener("click", async () => {
    const text = (ta.value || "").trim();
    if (!text) return;

    res.innerHTML = "";
    bar.classList.remove("hidden");
    btn.disabled = true;
    btn.textContent = "Выполняется анализ...";

    try {
      const r = await fetch("/diagnose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symptoms: text })
      });
      if (!r.ok) throw new Error("Server error");
      const data = await r.json();

      if (!data || !Array.isArray(data.diagnoses)) {
        res.innerHTML = `
          <div class="bg-red-900/30 border border-red-700 p-6 rounded-2xl">
            <div class="text-red-400 font-semibold text-lg">Неверный формат ответа сервера</div>
          </div>`;
        return;
      }

      const cards = data.diagnoses.slice(0,3).map(d => `
        <div class="p-4 rounded-xl bg-slate-950 border border-slate-800">
          <div class="flex items-center justify-between gap-3">
            <div class="text-sm text-slate-400">Rank #${escapeHtml(String(d.rank))}</div>
            <div class="text-sm font-semibold text-cyan-400">${escapeHtml(d.icd10_code)}</div>
          </div>
          <div class="mt-2 text-lg font-medium">${escapeHtml(d.diagnosis)}</div>
          <div class="mt-2 text-slate-300 whitespace-pre-line">${escapeHtml(d.explanation)}</div>
        </div>
      `).join("");

      res.innerHTML = `
        <h2 class="text-3xl font-semibold">Diagnostic Report</h2>
        <div class="grid gap-6">
          <div class="p-6 bg-slate-900 border border-slate-800 rounded-2xl">
            <div class="text-sm text-slate-400 mb-2">Top-3 ICD-10 hypotheses</div>
            <div class="space-y-4">${cards}</div>
          </div>
        </div>
      `;
    } catch (e) {
      res.innerHTML = `
        <div class="bg-red-900/30 border border-red-700 p-6 rounded-2xl">
          <div class="text-red-400 font-semibold text-lg">Ошибка подключения к серверу</div>
        </div>`;
    } finally {
      bar.classList.add("hidden");
      btn.disabled = false;
      btn.textContent = "Запустить диагностику";
    }
  });
</script>
</body>
</html>
        """.strip()
    )


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(req: DiagnoseRequest) -> DiagnoseResponse:
    query = normalize_text(req.symptoms or "")
    if not query or not protocol_texts:
        return DiagnoseResponse(diagnoses=[])

    TOPK = 80
    N = len(protocol_texts)

    ranked_word = []
    ranked_char = []

    if tfidf_word is not None:
        qv = vectorizer_word.transform([query])
        ranked_word = topk_indices(qv, tfidf_word, TOPK)

    if tfidf_char is not None:
        qv = vectorizer_char.transform([query])
        ranked_char = topk_indices(qv, tfidf_char, TOPK)

    ranks_word = compute_ranks(N, ranked_word) if ranked_word else [0] * N
    ranks_char = compute_ranks(N, ranked_char) if ranked_char else [0] * N

    # RRF fusion
    scores = [0.0] * N
    for i in range(N):
        if ranks_word[i] > 0:
            scores[i] += 1.0 / (60 + ranks_word[i])
        if ranks_char[i] > 0:
            scores[i] += 1.0 / (60 + ranks_char[i])

    top_docs = sorted(range(N), key=lambda i: scores[i], reverse=True)[:25]
    doc_scores = {i: scores[i] for i in top_docs}

    icd_ranked = score_icds(top_docs, doc_scores)[:3]

    diagnoses = []
    for r, (icd, _sc, best_doc) in enumerate(icd_ranked, start=1):
        diagnoses.append(
            Diagnosis(
                rank=r,
                diagnosis=f"Hypothesis: {icd}",
                icd10_code=icd,
                explanation=make_explanation(icd, best_doc),
            )
        )

    return DiagnoseResponse(diagnoses=diagnoses)

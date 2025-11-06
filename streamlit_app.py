# streamlit_app.py
# ---------------------------------------------------------------
# RivalLens Mini — SEO Automation Flow (Beautiful Streamlit UI)
# ---------------------------------------------------------------
# 1) Input up to 10 URLs
# 2) Fetch & parse content (cached + parallel)
# 3) Extract keywords (prefer multi-grams; filter junk)
# 4) Pull user Qs (Reddit/Quora via SerpAPI) — optional & rate-limited
# 5) Generate AI FAQ answers (OpenAI)
# 6) Recommend internal links across URLs
# 7) Pretty UI + export
# 8) ⚡ Fast mode, caching, and limited SerpAPI calls for speed
# ---------------------------------------------------------------

import os
import re
import json
import time
import html
import textwrap
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

# -------------------------
# Helpers & Config
# -------------------------
APP_TITLE = "RivalLens Mini — SEO Automation"
APP_SUBTITLE = "URLs → Keywords → User Questions → AI → Meta/Headings → Internal Links"
MAX_URLS = 10

DEFAULT_FAQ_COUNT = 5
USER_QUESTION_SOURCES = ["reddit.com", "quora.com"]

# Speed/limits
TEXT_TRIM_CHARS = 60000   # trim very long pages for faster n-grams/TF-IDF
FETCH_TTL = 60 * 60       # cache HTTP/SerpAPI for 1 hour

STOPWORDS = set(
    """
    a about above after again against all am an and any are aren't as at be because been before being below between both
    but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had
    hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd
    i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once
    only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such
    than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this
    those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where
    where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself
    yourselves
    """.split()
)

# Extra stopwords / verbs / fillers we don't want as keywords
EXTRA_STOPWORDS = {
    "can","could","should","would","may","might","must","also","etc","vs","—","–",
    "use","using","used","based","make","made","making","get","got","getting",
}
STOPWORDS |= EXTRA_STOPWORDS

# -------------------------
# Dataclasses
# -------------------------
@dataclass
class PageData:
    url: str
    title: str
    text: str
    keywords: List[str]
    questions: List[str]
    ai_faqs: List[Dict[str, str]]  # {question, answer}
    meta: Dict[str, str]            # {title, description, keywords}
    headings: Dict[str, List[str]]  # {h1: [..], h2: [..]}
    inner_links: List[Dict[str, str]]  # {source_url, anchor_text, target_url, reason}

# -------------------------
# UI Helpers
# -------------------------
def badge(text: str):
    st.markdown(
        f"""
        <span style="background:#eef2ff;color:#4338ca;padding:4px 10px;border-radius:999px;font-size:12px;border:1px solid #c7d2fe;">
        {html.escape(text)}
        </span>
        """,
        unsafe_allow_html=True,
    )

def section_title(title: str, subtitle: Optional[str] = None):
    st.markdown(f"<h2 style='margin-bottom:0'>{html.escape(title)}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f"<div style='color:#6b7280;margin-top:2px'>{html.escape(subtitle)}</div>",
            unsafe_allow_html=True,
        )

# -------------------------
# Cache helpers
# -------------------------
@st.cache_data(show_spinner=False, ttl=FETCH_TTL)
def cached_get(url: str, headers: Dict[str, str], timeout: int) -> str:
    return requests.get(url, headers=headers, timeout=timeout).text

@st.cache_data(show_spinner=False, ttl=FETCH_TTL)
def cached_serpapi(params_tuple: tuple) -> List[Dict]:
    url = "https://serpapi.com/search.json"
    params = dict(params_tuple)
    r = requests.get(url, params=params, timeout=15)
    if r.status_code == 200:
        return r.json().get("organic_results", [])
    return []

# -------------------------
# Core: Fetch & Parse
# -------------------------
def fetch_page(url: str, timeout: int = 15) -> Tuple[str, str]:
    """Return (title, visible_text) for a URL. Uses cache + trims long text."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36 RivalLensMini/1.0"
            )
        }
        html_text = cached_get(url, headers, timeout)
        # Prefer lxml if available for speed
        try:
            soup = BeautifulSoup(html_text, "lxml")
        except Exception:
            soup = BeautifulSoup(html_text, "html.parser")
        page_title = soup.title.text.strip() if soup.title else url
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text = " ".join(t.get_text(separator=" ", strip=True) for t in soup.find_all())
        text = re.sub(r"\s+", " ", text)
        if len(text) > TEXT_TRIM_CHARS:
            text = text[:TEXT_TRIM_CHARS]
        return page_title[:200], text
    except Exception:
        return url, ""

# -------------------------
# Keyword Extraction (prefer multi-grams; filter junk)
# -------------------------
def extract_keywords_basic(text: str, top_k: int = 12) -> List[str]:
    text = text.lower()[:TEXT_TRIM_CHARS]
    tokens = re.findall(r"[a-z][a-z\-]+", text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    # Build n-grams preferring multi-grams (3,2,1)
    grams: List[str] = []
    for n in [3, 2, 1]:
        for i in range(len(tokens) - n + 1):
            grams.append(" ".join(tokens[i : i + n]))

    # Frequency with filtering (skip grams containing stopwords)
    freq: Dict[str, int] = {}
    for g in grams:
        parts = g.split()
        if any(p in STOPWORDS for p in parts):
            continue
        freq[g] = freq.get(g, 0) + 1

    # Score: bonus for 2-3 grams; slight penalty for 1-grams
    scored = []
    for k, v in freq.items():
        n = len(k.split())
        bonus = 1.6 if n == 3 else (1.3 if n == 2 else 0.8)
        scored.append((k, v * bonus))
    scored.sort(key=lambda x: x[1], reverse=True)

    # De-duplicate, keep most informative
    selected: List[str] = []
    for k, _ in scored:
        if any(k in s or s in k for s in selected):
            continue
        selected.append(k)
        if len(selected) >= top_k:
            break

    # Ensure primary is multi-gram if possible
    multi = [s for s in selected if len(s.split()) >= 2]
    if multi:
        primary = multi[0]
        selected = [primary] + [s for s in selected if s != primary]

    return selected

# -------------------------
# User Questions via SerpAPI (Google) — fast & limited
# -------------------------
def search_questions_serpapi(query: str, serpapi_key: str, engine: str = "google", num: int = 10) -> List[Dict]:
    if not serpapi_key:
        return []
    params_tuple = tuple(sorted({
        "engine": engine,
        "q": query,
        "num": num,
        "api_key": serpapi_key,
        "hl": "en",
        "safe": "active",
    }.items()))
    try:
        return cached_serpapi(params_tuple)
    except Exception:
        return []

def collect_user_questions(keywords: List[str], serpapi_key: str, per_source: int = 5,
                           skip: bool = False, max_queries: int = 4) -> List[str]:
    """Find authentic questions from Reddit/Quora searches with fewer API calls."""
    if skip or not serpapi_key or not keywords:
        return []

    # Prefer multi-grams; limit keyword pool for speed
    kw_pool = sorted(keywords[:5], key=lambda k: -len(k.split()))[:2]

    queries: List[str] = []
    for src in USER_QUESTION_SOURCES:
        for kw in kw_pool:
            queries.extend([
                f"site:{src} {kw} what",
                f"site:{src} {kw} how",
                f"site:{src} {kw} best",
                f"site:{src} {kw} vs",
            ])
    queries = queries[:max_queries]

    questions: List[str] = []
    for q in queries:
        results = search_questions_serpapi(q, serpapi_key, num=5 if per_source <= 5 else 10)
        for res in results:
            title = (res.get("title") or "").strip()
            if not title:
                continue
            title = html.unescape(title)
            if title.endswith("?") or re.match(r"^(what|how|why|when|which|where|can|does|do)\b", title.lower()):
                if title not in questions:
                    questions.append(title)
        if len(questions) >= per_source * len(USER_QUESTION_SOURCES):
            break

    uniq: List[str] = []
    for q in questions:
        if q not in uniq:
            uniq.append(q)
    return uniq[: per_source * len(USER_QUESTION_SOURCES)]

# -------------------------
# OpenAI — text generation
# -------------------------
def generate_with_openai(prompt: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    if not api_key:
        return ""
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
        }
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=45)
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

# Demo fallback
def demo_generate(prompt: str) -> str:
    return textwrap.shorten(
        (
            "[DEMO OUTPUT] "
            "This is placeholder text because no API key was provided. "
            "Replace with real OpenAI output by adding your key in the sidebar.\n\n"
            f"Prompt: {prompt[:300]}"
        ),
        width=1000,
        placeholder="...",
    )

# -------------------------
# Internal Link Recommendations (TF-IDF Similarity)
# -------------------------
def recommend_internal_links(pages: List[PageData], top_n: int = 3) -> List[Dict[str, str]]:
    if len(pages) < 2:
        return []
    docs = [p.text for p in pages]
    tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words="english")
    X = tfidf.fit_transform(docs)
    sim = cosine_similarity(X)

    recs: List[Dict[str, str]] = []
    for i, src_page in enumerate(pages):
        sims = [(j, float(sim[i, j])) for j in range(len(pages)) if j != i]
        sims.sort(key=lambda x: x[1], reverse=True)
        for j, score in sims[:top_n]:
            tgt = pages[j]
            anchor = (src_page.keywords[:1] or [tgt.title.split("|")[0][:40]])[0]
            recs.append({
                "source_url": src_page.url,
                "anchor_text": anchor,
                "target_url": tgt.url,
                "reason": f"High topical similarity ({score:.2f}).",
            })
    return recs

# -------------------------
# AI Orchestration per Page
# -------------------------
def build_ai_prompts(page: PageData) -> Dict[str, str]:
    kw_line = ", ".join(page.keywords[:8])
    faq_block = "\n".join([f"- {q}" for q in page.questions[:DEFAULT_FAQ_COUNT]])

    prompts = {
        "faqs": f"""
You are an SEO & editorial expert. Given the page title and topic keywords below, write crisp, helpful answers (80-150 words) to the FAQs.

Page Title: {page.title}
Topic Keywords: {kw_line}
FAQs:
{faq_block}

Return as a JSON array of objects with keys: question, answer.
""".strip(),
        "meta": f"""
You are an SEO expert. Create meta title (<=60 chars), meta description (<=155 chars) and a comma-separated meta keywords string for the following page.

Page Title: {page.title}
Top Keywords: {kw_line}

Return JSON with keys: title, description, keywords.
""".strip(),
        # (Optional) Headings generator is kept but not used to overwrite
    }
    return prompts

def run_ai_generation(page: PageData, api_key: str, model: str, demo_mode: bool = False) -> PageData:
    gen = demo_generate if (demo_mode or not api_key) else (lambda p: generate_with_openai(p, api_key, model))
    prompts = build_ai_prompts(page)

    # FAQs (answers)
    faqs_raw = gen(prompts["faqs"]) or "[]"
    try:
        page.ai_faqs = json.loads(faqs_raw)
    except Exception:
        page.ai_faqs = [{"question": q, "answer": demo_generate("answer")[:200]} for q in page.questions[:DEFAULT_FAQ_COUNT]]

    # Meta (AI — filled if missing)
    meta_raw = gen(prompts["meta"]) or "{}"
    try:
        j = json.loads(meta_raw)
        page.meta = {
            "title": j.get("title") or page.meta.get("title") or page.title,
            "description": j.get("description") or page.meta.get("description") or "",
            "keywords": j.get("keywords") or page.meta.get("keywords") or ", ".join(page.keywords[:8]),
        }
    except Exception:
        if not page.meta:
            page.meta = {
                "title": page.title[:58],
                "description": demo_generate("meta description")[:150],
                "keywords": ", ".join(page.keywords[:8]),
            }
    return page

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")

    # Hero
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#f8fafc,#eef2ff);padding:28px;border-radius:24px;border:1px solid #e5e7eb;margin-bottom:14px;">
            <div style="font-size:28px;font-weight:700;">{html.escape(APP_TITLE)}</div>
            <div style="color:#6b7280;margin-top:6px">{html.escape(APP_SUBTITLE)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        badge("Tip: You can run in demo mode without keys")

        # Defaults from env/Secrets
        default_openai = os.getenv("OPENAI_API_KEY", "")
        default_serpapi = os.getenv("SERPAPI_KEY", "")

        openai_key_input = st.text_input("OpenAI API Key (optional)", type="password", value=default_openai)
        serpapi_key_input = st.text_input("SerpAPI Key (optional)", type="password", value=default_serpapi)

        model = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"], index=0)
        faq_count = st.slider("# of FAQs", min_value=3, max_value=8, value=DEFAULT_FAQ_COUNT)

        # Speed toggles
        fast_mode = st.toggle("⚡ Fast mode (optimize speed)", value=True)
        skip_serpapi = st.toggle("Skip Reddit/Quora questions (fastest)", value=False)
        limit_questions = st.slider("Max SerpAPI queries per URL", 2, 12, 4, help="Lower = faster")

        # Effective keys
        openai_key = openai_key_input or default_openai
        serpapi_key = serpapi_key_input or default_serpapi

        demo_mode = st.toggle("Demo mode (no external calls)", value=(openai_key == ""))
        st.caption("User questions are gathered from Reddit/Quora via Google (SerpAPI)")
        st.divider()
        st.caption("RivalLens Mini · v1.1 (fast)")

    # MAIN: Input
    section_title("1) Input URLs", "Add up to 10 URLs — we'll fetch, extract keywords & more")
    col1, col2 = st.columns([2, 1])
    with col1:
        urls_text = st.text_area(
            "Paste 1–10 URLs (one per line)",
            height=120,
            placeholder="https://example.com/blog/seo-guide\nhttps://example.com/blog/keyword-research",
        )
    with col2:
        st.write(""); st.write(""); st.write("")
        fetch_btn = st.button("Run Automation 🚀", type="primary", use_container_width=True)
        st.caption("We'll crawl pages, pull keywords, find user questions, and generate outputs")

    if fetch_btn:
        raw_urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        # Unique + limit
        seen = set()
        urls = []
        for u in raw_urls:
            if u not in seen:
                seen.add(u)
                urls.append(u)
            if len(urls) >= MAX_URLS:
                break

        if not urls:
            st.warning("Please add at least 1 URL.")
            st.stop()

        st.success(f"Processing {len(urls)} URL(s)...")

        pages: List[PageData] = []
        progress = st.progress(0)
        status = st.empty()

        # 1) Fetch pages in parallel for speed (30% of bar)
        status.info("Fetching pages…")
        pages_raw: Dict[str, Tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(urls))) as ex:
            futures = {ex.submit(fetch_page, u): u for u in urls}
            for n, fut in enumerate(as_completed(futures)):
                u = futures[fut]
                try:
                    title, text = fut.result()
                except Exception:
                    title, text = u, ""
                pages_raw[u] = (title, text)
                progress.progress(int(((n + 1) / max(1, len(urls))) * 30))

        # 2) Process each page (keywords, questions, AI) sequentially (remaining 70%)
        for idx, url in enumerate(urls):
            status.info(f"Analyzing: {url}")
            title, text = pages_raw.get(url, (url, ""))

            if not text:
                st.error(f"Could not fetch or parse content: {url}")
                text = ""

            kw = extract_keywords_basic(text, top_k=14)
            status.info(f"Extracted keywords for {url}")

            q = collect_user_questions(
                kw,
                serpapi_key,
                per_source=faq_count,
                skip=skip_serpapi,
                max_queries=(limit_questions if fast_mode else 12),
            )
            status.info(f"Found {len(q)} authentic user questions for {url}")

            page = PageData(
                url=url,
                title=title,
                text=text,
                keywords=kw,
                questions=q[:faq_count],
                ai_faqs=[],
                meta={},
                headings={},
                inner_links=[],
            )

            status.info(f"Generating AI outputs for {url}")
            page = run_ai_generation(page, openai_key, model, demo_mode=demo_mode)
            pages.append(page)

            progress.progress(30 + int(((idx + 1) / max(1, len(urls))) * 70))
            time.sleep(0.02 if fast_mode else 0.1)

        # Internal links
        status.info("Calculating internal link recommendations…")
        cross_links = recommend_internal_links(pages, top_n=3)
        for p in pages:
            p.inner_links = [rec for rec in cross_links if rec["source_url"] == p.url]

        status.success("Done! Review results below.")
        st.divider()

        # Results
        for p in pages:
            with st.container(border=True):
                st.markdown(f"### 🔗 {p.title}")
                st.caption(p.url)

                # Primary + chips
                primary_kw = p.keywords[0] if p.keywords else ""
                other_kws = [k for k in p.keywords[1:10]]
                st.markdown("**Primary keyword:** " + (f"`{primary_kw}`" if primary_kw else "_n/a_"))
                chips = " ".join(
                    [
                        f"<span style='background:#f1f5f9;padding:4px 8px;border-radius:999px;border:1px solid #e2e8f0;font-size:12px'>{html.escape(k)}</span>"
                        for k in other_kws
                    ]
                )
                st.markdown(chips, unsafe_allow_html=True)

                t1, t2, t3, t4 = st.tabs(["FAQs", "Meta & Headings", "Internal Links", "Raw Content"])

                with t1:
                    if p.questions:
                        st.markdown("**Top user questions (from Reddit/Quora searches):**")
                        for q_ in p.questions:
                            st.markdown(f"- {q_}")
                    else:
                        st.info("No questions found. Try adding a SerpAPI key or adjust keywords.")

                    st.markdown("**AI Answers:**")
                    if not p.ai_faqs:
                        st.write("No AI output (demo mode).")
                    else:
                        for item in p.ai_faqs[:faq_count]:
                            st.markdown(f"**Q:** {item.get('question','')}\n\n**A:** {item.get('answer','')}")

                with t2:
                    meta = p.meta or {}
                    st.markdown("#### Meta (AI-generated)")
                    st.write({k: meta.get(k, "") for k in ["title", "description", "keywords"]})

                    st.markdown("#### Headings (not scraped in this build)")
                    if p.headings:
                        st.write({"H1": p.headings.get("h1", []), "H2": p.headings.get("h2", [])})
                    else:
                        st.info("Headings not generated in this version.")

                with t3:
                    if p.inner_links:
                        df = pd.DataFrame(p.inner_links)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("Not enough pages for internal links.")

                with t4:
                    with st.expander("Show extracted plain text"):
                        st.write(p.text[:5000] + ("…" if len(p.text) > 5000 else ""))

        # Export
        st.divider()
        section_title("Export Results")
        all_rows: List[Dict[str, str]] = []
        for p in pages:
            faq_flat = json.dumps(p.ai_faqs, ensure_ascii=False)
            h1 = (p.headings.get("h1", [""]) or [""])[0] if p.headings else ""
            h2 = p.headings.get("h2", []) if p.headings else []
            recs = p.inner_links
            all_rows.append(
                {
                    "url": p.url,
                    "page_title": p.title,
                    "top_keywords": ", ".join(p.keywords[:10]),
                    "user_questions": "; ".join(p.questions[:faq_count]),
                    "ai_faqs": faq_flat,
                    "meta_title": p.meta.get("title", ""),
                    "meta_description": p.meta.get("description", ""),
                    "meta_keywords": p.meta.get("keywords", ""),
                    "h1": h1,
                    "h2": "; ".join(h2),
                    "internal_links": json.dumps(recs, ensure_ascii=False),
                }
            )
        export_df = pd.DataFrame(all_rows)
        st.dataframe(export_df, use_container_width=True)

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        json_bytes = json.dumps(all_rows, ensure_ascii=False, indent=2).encode("utf-8")

        colA, colB = st.columns(2)
        with colA:
            st.download_button("Download CSV", data=csv_bytes, file_name="rivallens_mini_results.csv", mime="text/csv")
        with colB:
            st.download_button("Download JSON", data=json_bytes, file_name="rivallens_mini_results.json", mime="application/json")

    # Footer
    st.markdown(
        """
        <hr/>
        <div style="color:#6b7280;font-size:12px">Tip: Add API keys in the sidebar for live AI and question sourcing. Use ⚡ Fast mode & Skip questions for speed.</div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()

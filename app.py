"""AI Micro-Content & Social Media Repurposer
Turns long-form content into an X/Twitter thread, a LinkedIn post and a
TikTok/Reels script using the Groq API.

Run with:  streamlit run app.py
"""

import os
import re
from datetime import datetime

import streamlit as st

try:
    from groq import Groq
except ImportError:  # handled gracefully in the UI
    Groq = None

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AI Content Repurposer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]
TONES = ["Professional", "Casual", "Witty", "Inspirational"]
LENGTHS = ["Standard", "Short & Punchy", "Detailed"]

MAX_INPUT_CHARS = 30_000
MIN_INPUT_CHARS = 80

HEADERS = {
    "twitter": "### 🧵 X/Twitter Thread",
    "linkedin": "### 💼 LinkedIn Post",
    "tiktok": "### 🎬 TikTok / Reels Script",
}

LENGTH_GUIDANCE = {
    "Standard": (
        "Balanced depth. Thread: 6-8 tweets. LinkedIn: 150-250 words. "
        "Video script: about 45-60 seconds."
    ),
    "Short & Punchy": (
        "Ultra concise and high-impact. Thread: 4-5 tweets. LinkedIn: 80-130 words. "
        "Video script: about 20-30 seconds."
    ),
    "Detailed": (
        "In-depth and rich in insight. Thread: 9-12 tweets. LinkedIn: 300-450 words. "
        "Video script: about 60-90 seconds."
    ),
}

TONE_GUIDANCE = {
    "Professional": "Polished, credible, authoritative, clear. No slang.",
    "Casual": "Conversational, friendly, like talking to a smart friend. Contractions welcome.",
    "Witty": "Clever, playful, sharp humor and wordplay, without being cringe or off-topic.",
    "Inspirational": "Uplifting, motivating, story-driven, emotionally resonant.",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_api_key(sidebar_key: str) -> str:
    """Resolve the Groq API key: sidebar input > st.secrets > environment."""
    if sidebar_key and sidebar_key.strip():
        return sidebar_key.strip()

    try:
        secret = st.secrets.get("GROQ_API_KEY", "")
        if secret:
            return str(secret).strip()
    except Exception:
        # No secrets.toml present or unreadable - fall through to env var.
        pass

    return os.environ.get("GROQ_API_KEY", "").strip()


def build_messages(source: str, tone: str, length: str) -> list:
    system_prompt = (
        "You are an elite social media manager and content strategist who has grown "
        "multiple accounts to millions of followers. You specialise in repurposing "
        "long-form content into viral, platform-native assets. You never invent facts, "
        "statistics or quotes that are not supported by the source material."
    )

    user_prompt = f"""Repurpose the SOURCE CONTENT below into exactly 3 distinct, platform-native assets.

TONE OF VOICE: {tone} - {TONE_GUIDANCE[tone]}
LENGTH / DEPTH: {length} - {LENGTH_GUIDANCE[length]}

OUTPUT FORMAT (follow EXACTLY, use these markdown headers verbatim, in this order, with no text before the first header and no text after the last section):

{HEADERS['twitter']}
- Start with a scroll-stopping hook tweet.
- Number each tweet like "1/", "2/", etc. Keep each under 280 characters.
- Each tweet delivers one clear idea; end with a strong CTA tweet.
- Add at most 2 relevant hashtags, only in the final tweet.

{HEADERS['linkedin']}
- Open with a compelling one-line hook, then use short paragraphs and generous line breaks.
- Share insights, a lesson or a story drawn from the source; end with an engaging question or CTA.
- Add 3-5 relevant hashtags at the end.

{HEADERS['tiktok']}
- Structure as: HOOK (first 3 seconds), BODY (key points), CTA.
- For each beat include "Voiceover:" and "On-screen text / Visual:" lines.
- Include a suggested caption and 3-5 hashtags at the end.

RULES:
- Produce each asset as standalone content that does not require reading the source.
- Make each asset feel native to its platform, not a copy of the others.
- Use ONLY the three headers above; do not add any other level-3 headers.

SOURCE CONTENT:
\"\"\"
{source}
\"\"\"
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_content(api_key: str, model: str, source: str, tone: str, length: str) -> str:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(source, tone, length),
        temperature=0.7,
    )
    content = response.choices[0].message.content if response.choices else ""
    return (content or "").strip()


def parse_sections(text: str) -> dict:
    """Split the model output into platform sections.

    Tolerant of header-level or emoji variations. Returns a dict with keys
    'twitter', 'linkedin', 'tiktok' (missing sections map to empty strings).
    """
    result = {"twitter": "", "linkedin": "", "tiktok": ""}
    patterns = {
        "twitter": r"X\s*/\s*Twitter|Twitter",
        "linkedin": r"LinkedIn",
        "tiktok": r"TikTok|Reels",
    }

    header_re = re.compile(r"^[ \t]*#{1,6}[ \t]*(.+?)[ \t]*$", re.MULTILINE)
    found = []  # (key, header_start, content_start)
    seen = set()
    for m in header_re.finditer(text):
        title = m.group(1)
        for key, pat in patterns.items():
            if key not in seen and re.search(pat, title, re.IGNORECASE):
                found.append((key, m.start(), m.end()))
                seen.add(key)
                break

    found.sort(key=lambda item: item[1])
    for i, (key, _start, content_start) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else len(text)
        result[key] = text[content_start:end].strip()

    return result


def build_markdown_export(sections: dict, tone: str, length: str, model: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "# Social Media Content Pack",
        f"_Generated {stamp} | Model: {model} | Tone: {tone} | Length: {length}_",
        "",
    ]
    for key in ("twitter", "linkedin", "tiktok"):
        parts.append(HEADERS[key])
        parts.append("")
        parts.append(sections.get(key) or "_No content generated._")
        parts.append("")
    return "\n".join(parts)


def render_tab(label: str, content: str, raw_fallback: str, file_stub: str) -> None:
    if not content:
        st.warning(
            f"Couldn't isolate the {label} section automatically. "
            "Showing the full raw output below."
        )
        content = raw_fallback

    st.markdown("**Preview**")
    st.markdown(content)
    st.divider()
    st.markdown("**Copy-ready text** (use the copy icon in the top-right of the box)")
    st.code(content, language="markdown", wrap_lines=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("🔑 API Key")
    sidebar_key = st.text_input(
        "Groq API key (optional override)",
        type="password",
        placeholder="gsk_...",
        help=(
            "Leave blank to use st.secrets['GROQ_API_KEY'] or the GROQ_API_KEY "
            "environment variable. Get a key at console.groq.com."
        ),
    )
    api_key = resolve_api_key(sidebar_key)
    if api_key:
        st.success("API key detected", icon="✅")
    else:
        st.warning("No API key found yet", icon="⚠️")

    st.subheader("🤖 Model")
    model = st.selectbox("Groq model", MODELS, index=0)
    if model == "mixtral-8x7b-32768":
        st.caption(
            "Note: Groq has deprecated this model on some accounts. "
            "If it errors, switch to a Llama model."
        )

    st.subheader("🎨 Style")
    tone = st.selectbox("Tone of voice", TONES, index=0)
    length = st.selectbox("Length / depth", LENGTHS, index=0)

    st.divider()
    st.caption("Built with Streamlit + Groq")

# --------------------------------------------------------------------------- #
# Main interface
# --------------------------------------------------------------------------- #
st.title("🚀 AI Micro-Content & Social Media Repurposer")
st.subheader(
    "Turn one blog post, essay or transcript into a Twitter/X thread, "
    "a LinkedIn post and a TikTok/Reels script in seconds."
)
st.write(
    "Paste your long-form content below, choose a tone and length in the sidebar, "
    "and let the AI craft platform-native assets for you."
)

source_text = st.text_area(
    "Source content",
    height=320,
    placeholder="Paste your blog post, essay, newsletter or video transcript here...",
)
char_count = len(source_text.strip())
st.caption(f"{char_count:,} characters / {MAX_INPUT_CHARS:,} max")

generate = st.button("✨ Generate Social Content", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None

if generate:
    st.session_state.result = None
    text = source_text.strip()

    if Groq is None:
        st.error("The `groq` package is not installed. Run `pip install groq` and restart the app.")
    elif not api_key:
        st.error(
            "No Groq API key found. Add it in the sidebar, set `GROQ_API_KEY` in "
            "`.streamlit/secrets.toml`, or export it as an environment variable."
        )
    elif not text:
        st.warning("Please paste some source content first.")
    elif len(text) < MIN_INPUT_CHARS:
        st.warning(
            f"That's quite short ({len(text)} characters). "
            f"Please provide at least {MIN_INPUT_CHARS} characters for good results."
        )
    else:
        if len(text) > MAX_INPUT_CHARS:
            st.warning(
                f"Your content exceeds {MAX_INPUT_CHARS:,} characters, so it was truncated "
                "to fit the model's context window."
            )
            text = text[:MAX_INPUT_CHARS]

        try:
            with st.spinner("Crafting your viral content..."):
                raw = generate_content(api_key, model, text, tone, length)

            if not raw:
                st.error("The model returned an empty response. Please try again.")
            else:
                sections = parse_sections(raw)
                st.session_state.result = {
                    "raw": raw,
                    "sections": sections,
                    "tone": tone,
                    "length": length,
                    "model": model,
                }
        except Exception as exc:  # noqa: BLE001 - surface any API problem nicely
            msg = str(exc)
            low = msg.lower()
            if "401" in msg or "invalid api key" in low or "authentication" in low:
                st.error("Authentication failed. Please check that your Groq API key is valid.")
            elif "429" in msg or "rate limit" in low:
                st.error("Rate limit reached. Please wait a moment and try again.")
            elif "model" in low and ("not found" in low or "decommissioned" in low or "deprecated" in low):
                st.error(
                    f"The model `{model}` isn't available on your account. "
                    "Please pick a different model in the sidebar."
                )
            elif "connection" in low or "timeout" in low or "timed out" in low:
                st.error("Couldn't reach the Groq API. Check your internet connection and try again.")
            else:
                st.error(f"Something went wrong while generating content: {msg}")

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
result = st.session_state.result
if result:
    st.success("Your content is ready! 🎉")

    sections = result["sections"]
    raw = result["raw"]

    tab_x, tab_li, tab_tt = st.tabs(["🧵 X/Twitter Thread", "💼 LinkedIn Post", "🎬 TikTok / Reels"])
    with tab_x:
        render_tab("X/Twitter", sections["twitter"], raw, "twitter")
    with tab_li:
        render_tab("LinkedIn", sections["linkedin"], raw, "linkedin")
    with tab_tt:
        render_tab("TikTok / Reels", sections["tiktok"], raw, "tiktok")

    st.divider()
    export_md = build_markdown_export(sections, result["tone"], result["length"], result["model"])
    st.download_button(
        "⬇️ Download all as Markdown (.md)",
        data=export_md,
        file_name=f"social_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

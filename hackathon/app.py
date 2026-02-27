import streamlit as st
import streamlit.components.v1 as components
import os
import json
from datetime import datetime
from rag_pipeline import TrainingBot
from utils import text_to_speech, save_chat_history, get_language_code

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Training Agent",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

# ─── CSS + particle background via JS injection ────────────────────────────────
def inject_css(css: str):
    components.html(
        f"<script>window.parent.document.head.insertAdjacentHTML('beforeend',`<style>{css}</style>`)</script>",
        height=0, width=0,
    )

# Particle canvas injected into main page
PARTICLE_JS = """
<canvas id="particles" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none"></canvas>
<script>
(function(){
  var canvas = window.parent.document.getElementById('particles');
  if(!canvas){
    canvas = window.parent.document.createElement('canvas');
    canvas.id='particles';
    canvas.style.cssText='position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none';
    window.parent.document.body.appendChild(canvas);
  }
  var ctx = canvas.getContext('2d');
  canvas.width = window.parent.innerWidth;
  canvas.height = window.parent.innerHeight;
  var dots = Array.from({length:60},()=>({
    x:Math.random()*canvas.width,
    y:Math.random()*canvas.height,
    r:Math.random()*2+0.5,
    dx:(Math.random()-0.5)*0.3,
    dy:(Math.random()-0.5)*0.3,
    o:Math.random()*0.5+0.2
  }));
  function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    dots.forEach(d=>{
      ctx.beginPath();
      ctx.arc(d.x,d.y,d.r,0,Math.PI*2);
      ctx.fillStyle='rgba(0,180,255,'+d.o+')';
      ctx.fill();
      d.x+=d.dx; d.y+=d.dy;
      if(d.x<0||d.x>canvas.width) d.dx*=-1;
      if(d.y<0||d.y>canvas.height) d.dy*=-1;
    });
    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
"""

SCIFI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Pure black background */
.stApp { background: #000008 !important; }
.block-container { background: transparent !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: rgba(0,10,30,0.95) !important;
  border-right: 1px solid rgba(0,180,255,0.2) !important;
  backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * { color: #A0C8E8 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #00D4FF !important; }

[data-testid="stSidebar"] .stSelectbox > div > div {
  background: rgba(0,180,255,0.07) !important;
  border: 1px solid rgba(0,180,255,0.3) !important;
  border-radius: 10px !important;
  color: #A0C8E8 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: rgba(0,180,255,0.05) !important;
  border: 1.5px dashed rgba(0,180,255,0.35) !important;
  border-radius: 12px !important;
}

/* ── Buttons ── */
.stButton > button {
  background: transparent !important;
  color: #00D4FF !important;
  border: 1.5px solid rgba(0,180,255,0.5) !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.83rem !important;
  padding: 9px 18px !important;
  width: 100% !important;
  transition: all 0.25s ease !important;
  letter-spacing: 0.5px;
  box-shadow: 0 0 10px rgba(0,180,255,0.15) !important;
}
.stButton > button:hover {
  background: rgba(0,180,255,0.12) !important;
  box-shadow: 0 0 22px rgba(0,180,255,0.4) !important;
  border-color: #00D4FF !important;
  transform: translateY(-1px) !important;
}
.stDownloadButton > button {
  background: transparent !important;
  color: #00D4FF !important;
  border: 1px solid rgba(0,180,255,0.3) !important;
  border-radius: 8px !important;
  box-shadow: none !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] { background: transparent !important; padding: 1px 0 !important; }

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p {
  background: linear-gradient(135deg,rgba(0,130,200,0.3),rgba(0,80,160,0.3));
  border: 1px solid rgba(0,180,255,0.35);
  color: #C8E8FF !important;
  border-radius: 18px 18px 4px 18px;
  padding: 10px 16px;
  display: inline-block;
  max-width: 75%;
  font-size: 0.88rem;
  line-height: 1.55;
  box-shadow: 0 0 16px rgba(0,140,255,0.2);
  backdrop-filter: blur(8px);
}
/* Bot bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown p {
  background: linear-gradient(135deg,rgba(0,40,80,0.6),rgba(0,20,50,0.6));
  border: 1px solid rgba(0,180,255,0.2);
  color: #A0C8E8 !important;
  border-radius: 18px 18px 18px 4px;
  padding: 10px 16px;
  display: inline-block;
  max-width: 80%;
  font-size: 0.88rem;
  line-height: 1.55;
  backdrop-filter: blur(8px);
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
  background: rgba(0,20,50,0.8) !important;
  border-radius: 30px !important;
  border: 1.5px solid rgba(0,180,255,0.4) !important;
  box-shadow: 0 0 20px rgba(0,140,255,0.25), inset 0 0 20px rgba(0,0,0,0.5) !important;
  backdrop-filter: blur(12px) !important;
}
[data-testid="stChatInputTextArea"] { color: #A0C8E8 !important; font-size: 0.9rem !important; }
[data-testid="stChatInputTextArea"]::placeholder { color: #2A5080 !important; }

/* ── Alerts ── */
.stAlert { background: rgba(0,40,80,0.6) !important; border-radius: 10px !important; font-size: 0.82rem !important; border-color: rgba(0,180,255,0.3) !important; }

/* ── Divider ── */
hr { border-color: rgba(0,180,255,0.12) !important; }

/* ── Spin ── */
.stSpinner > div { border-top-color: #00D4FF !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,180,255,0.25); border-radius: 4px; }

/* ── Audio ── */
audio { border-radius: 10px; width:100%; margin-top:6px; filter: invert(0.9) hue-rotate(170deg) brightness(0.8); }
"""

inject_css(SCIFI_CSS)

# Inject particle dots
components.html(PARTICLE_JS, height=0)

# ─── Autoplay helper (native Streamlit, no CSP issues) ───────────────────────
def play_audio(audio_bio):
    """Play audio using Streamlit's native autoplay — works without any browser CSP workarounds."""
    audio_bio.seek(0)
    st.audio(audio_bio, format="audio/mp3", autoplay=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "bot" not in st.session_state:
    with st.spinner("Booting AI Agent…"):
        st.session_state.bot = TrainingBot()
if "history" not in st.session_state:
    st.session_state.history = []
if "summary" not in st.session_state:
    st.session_state.summary = None

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # Glowing logo header
    st.markdown("""
    <div style="text-align:center;padding:24px 10px 18px;border-bottom:1px solid rgba(0,180,255,0.15);margin-bottom:20px">
      <div style="width:70px;height:70px;margin:0 auto 12px;border-radius:50%;
                  background:radial-gradient(circle,rgba(0,200,255,0.4) 0%,rgba(0,80,160,0.3) 50%,transparent 70%);
                  border:2px solid rgba(0,200,255,0.6);
                  box-shadow:0 0 30px rgba(0,200,255,0.5),0 0 60px rgba(0,100,200,0.3);
                  display:flex;align-items:center;justify-content:center;font-size:1.8rem">
        🤖
      </div>
      <div style="font-family:'Orbitron',monospace;font-size:0.95rem;font-weight:700;
                  color:#00D4FF;letter-spacing:2px;text-shadow:0 0 10px rgba(0,212,255,0.7)">
        AI AGENT
      </div>
      <div style="font-size:0.7rem;color:#2A6080;letter-spacing:1px;margin-top:3px">
        VERNACULAR TRAINING SYSTEM
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Status indicator
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;background:rgba(0,180,255,0.06);
                border:1px solid rgba(0,180,255,0.15);border-radius:10px;
                padding:8px 12px;margin-bottom:16px">
      <div style="width:8px;height:8px;border-radius:50%;background:#00FF88;
                  box-shadow:0 0 8px #00FF88;animation:pulse 2s infinite"></div>
      <span style="font-size:0.78rem;color:#00FF88;letter-spacing:0.5px">ONLINE · READY</span>
    </div>
    <style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}</style>
    """, unsafe_allow_html=True)

    # Language
    st.markdown('<div style="font-size:0.68rem;font-weight:700;letter-spacing:1.5px;color:#1A4060;text-transform:uppercase;margin-bottom:6px">◈ Language Module</div>', unsafe_allow_html=True)
    lang_choice = st.selectbox("", ["English", "Hindi", "Telugu"], label_visibility="collapsed")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # PDF Upload
    st.markdown('<div style="font-size:0.68rem;font-weight:700;letter-spacing:1.5px;color:#1A4060;text-transform:uppercase;margin-bottom:6px">◈ Knowledge Base</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if "last_lang" in st.session_state and st.session_state.last_lang != lang_choice:
            st.session_state.summary = None
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            with open("temp_policy.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Indexing document…"):
                status = st.session_state.bot.ingest_pdf("temp_policy.pdf")
                st.session_state.last_uploaded = uploaded_file.name
                st.session_state.last_lang = lang_choice
                st.session_state.summary = None
                st.success(status)

        if not st.session_state.summary:
            if st.button("⚡ Generate Summary"):
                with st.spinner("Processing…"):
                    st.session_state.summary = st.session_state.bot.generate_summary(lang_choice)
                    st.session_state.last_lang = lang_choice
                    st.rerun()
        elif st.session_state.last_lang != lang_choice:
            if st.button("↺ Refresh Summary"):
                with st.spinner("Updating…"):
                    st.session_state.summary = st.session_state.bot.generate_summary(lang_choice)
                    st.session_state.last_lang = lang_choice
                    st.rerun()

    # Summary
    if st.session_state.summary:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.68rem;font-weight:700;letter-spacing:1.5px;color:#1A4060;text-transform:uppercase;margin-bottom:6px">◈ Document Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:rgba(0,180,255,0.05);border:1px solid rgba(0,180,255,0.2);'
            f'border-left:3px solid #00D4FF;border-radius:8px;padding:12px 14px;'
            f'font-size:0.8rem;color:#6A9EC0;line-height:1.65">'
            f'{st.session_state.summary}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(0,180,255,0.3),transparent);margin:0 -8px 16px"></div>', unsafe_allow_html=True)

    if st.button("⬡ Clear Session"):
        st.session_state.history = []
        st.session_state.summary = None
        if os.path.exists("chat_history.json"):
            os.remove("chat_history.json")
        st.rerun()

    if st.session_state.history:
        chat_json = json.dumps(st.session_state.history, indent=4)
        st.download_button("↓ Export Log", data=chat_json,
                           file_name="chat_history.json", mime="application/json")

# ─── Main layout: center chat | right panel ───────────────────────────────────
col_chat, col_info = st.columns([3, 1], gap="medium")

with col_chat:
    # ── Glowing header card ──
    st.markdown(f"""
    <div style="position:relative;text-align:center;padding:28px 24px 22px;
                background:linear-gradient(180deg,rgba(0,20,50,0.9) 0%,rgba(0,8,25,0.95) 100%);
                border:1px solid rgba(0,200,255,0.25);border-radius:20px;
                box-shadow:0 0 40px rgba(0,140,255,0.15),inset 0 0 40px rgba(0,0,20,0.5);
                margin-bottom:16px;backdrop-filter:blur(20px)">
      <!-- Glow orb -->
      <div style="width:80px;height:80px;margin:0 auto 14px;border-radius:50%;
                  background:radial-gradient(circle,rgba(0,220,255,0.5) 0%,rgba(0,100,200,0.3) 40%,transparent 70%);
                  border:2px solid rgba(0,220,255,0.5);
                  box-shadow:0 0 40px rgba(0,180,255,0.6),0 0 80px rgba(0,100,200,0.4);
                  display:flex;align-items:center;justify-content:center;font-size:2rem">
        🤖
      </div>
      <div style="font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:900;
                  background:linear-gradient(90deg,#00B4FF,#00FFD4,#00B4FF);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  letter-spacing:3px;text-shadow:none;margin-bottom:6px">
        AI TRAINING AGENT
      </div>
      <div style="font-size:0.75rem;color:#2A5A80;letter-spacing:2px">
        VERNACULAR EMPLOYEE SYSTEM · {lang_choice.upper()}
      </div>
      <!-- Decorative dots -->
      <div style="position:absolute;top:12px;left:16px;display:flex;gap:5px">
        <div style="width:8px;height:8px;border-radius:50%;background:#FF4466;opacity:0.7"></div>
        <div style="width:8px;height:8px;border-radius:50%;background:#FFB800;opacity:0.7"></div>
        <div style="width:8px;height:8px;border-radius:50%;background:#00D4FF;opacity:0.7"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chat messages ──
    if not st.session_state.history:
        st.markdown("""
        <div style="background:linear-gradient(180deg,rgba(0,15,40,0.8),rgba(0,5,20,0.9));
                    border:1px solid rgba(0,180,255,0.15);border-radius:16px;
                    padding:50px 30px;text-align:center;margin-bottom:14px;backdrop-filter:blur(10px)">
          <div style="font-size:2.8rem;margin-bottom:12px;filter:drop-shadow(0 0 12px rgba(0,200,255,0.6))">�</div>
          <div style="font-family:'Orbitron',monospace;font-size:0.85rem;color:#00A0C0;
                      letter-spacing:1px;margin-bottom:8px">AWAITING INPUT</div>
          <div style="font-size:0.8rem;color:#1A4060;max-width:320px;margin:0 auto;line-height:1.7">
            Upload a policy document from the sidebar panel, then initiate your query below.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for chat in st.session_state.history:
            with st.chat_message(chat["role"]):
                st.write(chat["content"])

    # ── Chat input ──
    if prompt := st.chat_input("SEND QUERY →"):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing…"):
                answer = st.session_state.bot.get_answer(prompt, lang_choice)
                st.write(answer)

                if not answer.startswith(("⚠️", "❌", "Please upload")):
                    try:
                        lang_code = get_language_code(lang_choice)
                        audio_bio = text_to_speech(answer, lang_code)
                        play_audio(audio_bio)
                    except Exception as tts_err:
                        st.warning(f"🔇 Voice unavailable: {tts_err}")

                st.session_state.history.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": str(datetime.now()),
                })
                save_chat_history(st.session_state.history)

# ─── Right info panel ─────────────────────────────────────────────────────────
with col_info:
    # System status
    q_count = len([h for h in st.session_state.history if h["role"] == "user"])
    st.markdown(f"""
    <div style="background:linear-gradient(180deg,rgba(0,15,40,0.9),rgba(0,5,20,0.95));
                border:1px solid rgba(0,180,255,0.18);border-radius:14px;
                padding:16px;margin-bottom:12px;backdrop-filter:blur(12px)">
      <!-- Browser dots -->
      <div style="display:flex;gap:5px;margin-bottom:12px">
        <div style="width:7px;height:7px;border-radius:50%;background:#FF4466;opacity:0.8"></div>
        <div style="width:7px;height:7px;border-radius:50%;background:#FFB800;opacity:0.8"></div>
        <div style="width:7px;height:7px;border-radius:50%;background:#00D4FF;opacity:0.8"></div>
      </div>
      <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.5px;color:#1A4060;
                  text-transform:uppercase;margin-bottom:10px">SYSTEM STATUS</div>
      <div style="display:flex;gap:8px;margin-bottom:8px">
        <div style="flex:1;background:rgba(0,180,255,0.07);border:1px solid rgba(0,180,255,0.15);
                    border-radius:8px;padding:10px;text-align:center">
          <div style="font-family:'Orbitron',monospace;font-size:1.4rem;color:#00D4FF;font-weight:700">{q_count}</div>
          <div style="font-size:0.65rem;color:#1A4060;margin-top:2px;letter-spacing:0.5px">QUERIES</div>
        </div>
        <div style="flex:1;background:rgba(0,255,136,0.07);border:1px solid rgba(0,255,136,0.2);
                    border-radius:8px;padding:10px;text-align:center">
          <div style="font-size:1.2rem;color:#00FF88">{"●" if uploaded_file else "○"}</div>
          <div style="font-size:0.65rem;color:#1A4060;margin-top:2px;letter-spacing:0.5px">DOC</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Models card
    st.markdown("""
    <div style="background:linear-gradient(180deg,rgba(0,15,40,0.9),rgba(0,5,20,0.95));
                border:1px solid rgba(0,180,255,0.18);border-radius:14px;
                padding:16px;margin-bottom:12px;backdrop-filter:blur(12px)">
      <div style="display:flex;gap:5px;margin-bottom:12px">
        <div style="width:7px;height:7px;border-radius:50%;background:#FF4466;opacity:0.8"></div>
        <div style="width:7px;height:7px;border-radius:50%;background:#FFB800;opacity:0.8"></div>
        <div style="width:7px;height:7px;border-radius:50%;background:#00D4FF;opacity:0.8"></div>
      </div>
      <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.5px;color:#1A4060;
                  text-transform:uppercase;margin-bottom:10px">AI MODULES</div>
      <div style="display:flex;flex-direction:column;gap:7px">
        <div style="background:rgba(0,180,255,0.06);border:1px solid rgba(0,180,255,0.15);
                    border-radius:8px;padding:8px 10px;display:flex;align-items:center;gap:8px">
          <span style="font-size:0.85rem">🧠</span>
          <div>
            <div style="font-size:0.75rem;color:#00D4FF;font-weight:600">Llama 3</div>
            <div style="font-size:0.65rem;color:#1A4060">Primary LLM</div>
          </div>
          <div style="margin-left:auto;width:6px;height:6px;border-radius:50%;
                      background:#00FF88;box-shadow:0 0 6px #00FF88"></div>
        </div>
        <div style="background:rgba(0,180,255,0.06);border:1px solid rgba(0,180,255,0.15);
                    border-radius:8px;padding:8px 10px;display:flex;align-items:center;gap:8px">
          <span style="font-size:0.85rem">⚡</span>
          <div>
            <div style="font-size:0.75rem;color:#00D4FF;font-weight:600">Qwen 2.5</div>
            <div style="font-size:0.65rem;color:#1A4060">Fallback LLM</div>
          </div>
          <div style="margin-left:auto;width:6px;height:6px;border-radius:50%;
                      background:#00FF88;box-shadow:0 0 6px #00FF88"></div>
        </div>
        <div style="background:rgba(0,180,255,0.06);border:1px solid rgba(0,180,255,0.15);
                    border-radius:8px;padding:8px 10px;display:flex;align-items:center;gap:8px">
          <span style="font-size:0.85rem">�</span>
          <div>
            <div style="font-size:0.75rem;color:#00D4FF;font-weight:600">TTS Engine</div>
            <div style="font-size:0.65rem;color:#1A4060">Auto voice</div>
          </div>
          <div style="margin-left:auto;width:6px;height:6px;border-radius:50%;
                      background:#00FF88;box-shadow:0 0 6px #00FF88"></div>
        </div>
        <div style="background:rgba(0,180,255,0.06);border:1px solid rgba(0,180,255,0.15);
                    border-radius:8px;padding:8px 10px;display:flex;align-items:center;gap:8px">
          <span style="font-size:0.85rem">🔎</span>
          <div>
            <div style="font-size:0.75rem;color:#00D4FF;font-weight:600">FAISS RAG</div>
            <div style="font-size:0.65rem;color:#1A4060">Vector search</div>
          </div>
          <div style="margin-left:auto;width:6px;height:6px;border-radius:50%;
                      background:#00FF88;box-shadow:0 0 6px #00FF88"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Recent queries
    user_qs = [h["content"] for h in st.session_state.history if h["role"] == "user"][-3:]
    if user_qs:
        st.markdown("""
        <div style="background:linear-gradient(180deg,rgba(0,15,40,0.9),rgba(0,5,20,0.95));
                    border:1px solid rgba(0,180,255,0.18);border-radius:14px;
                    padding:16px;backdrop-filter:blur(12px)">
          <div style="display:flex;gap:5px;margin-bottom:12px">
            <div style="width:7px;height:7px;border-radius:50%;background:#FF4466;opacity:0.8"></div>
            <div style="width:7px;height:7px;border-radius:50%;background:#FFB800;opacity:0.8"></div>
            <div style="width:7px;height:7px;border-radius:50%;background:#00D4FF;opacity:0.8"></div>
          </div>
          <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.5px;color:#1A4060;
                      text-transform:uppercase;margin-bottom:10px">QUERY LOG</div>
        """, unsafe_allow_html=True)
        for q in reversed(user_qs):
            st.markdown(f"""
          <div style="background:rgba(0,180,255,0.05);border-left:2px solid rgba(0,180,255,0.4);
                      border-radius:0 8px 8px 0;padding:7px 10px;margin-bottom:6px;
                      font-size:0.75rem;color:#4A7A9A;line-height:1.45">
            {q[:55]}{"…" if len(q)>55 else ""}
          </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:linear-gradient(180deg,rgba(0,15,40,0.9),rgba(0,5,20,0.95));
                    border:1px solid rgba(0,180,255,0.18);border-radius:14px;
                    padding:16px;backdrop-filter:blur(12px)">
          <div style="display:flex;gap:5px;margin-bottom:12px">
            <div style="width:7px;height:7px;border-radius:50%;background:#FF4466;opacity:0.8"></div>
            <div style="width:7px;height:7px;border-radius:50%;background:#FFB800;opacity:0.8"></div>
            <div style="width:7px;height:7px;border-radius:50%;background:#00D4FF;opacity:0.8"></div>
          </div>
          <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.5px;color:#1A4060;
                      text-transform:uppercase;margin-bottom:10px">QUERY LOG</div>
          <div style="font-size:0.78rem;color:#1A3050;text-align:center;padding:10px 0;
                      font-family:'Orbitron',monospace;letter-spacing:1px">NO DATA</div>
        </div>
        """, unsafe_allow_html=True)

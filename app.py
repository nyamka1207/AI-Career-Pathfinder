"""
🚀 AI Career Pathfinder
A Streamlit app that trains a RandomForestClassifier on synthetic career data
and predicts the most suitable career path based on user skills.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Pathfinder",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CUSTOM CSS  — deep-space blue + electric violet palette
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0b0f1a 0%, #111827 60%, #0f172a 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    border-right: 1px solid #312e81;
}
[data-testid="stSidebar"] * { color: #c7d2fe !important; }
[data-testid="stSidebar"] .stSlider > label { color: #a5b4fc !important; font-weight: 600; }

/* Main title */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid #4338ca;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(99,102,241,0.15);
}
.metric-label {
    color: #a5b4fc;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #e0e7ff;
}
.metric-icon { font-size: 1.4rem; margin-bottom: 0.2rem; }

/* Prediction result card */
.result-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #2e1065 100%);
    border: 2px solid #7c3aed;
    border-radius: 20px;
    padding: 1.8rem 2rem;
    box-shadow: 0 8px 32px rgba(124,58,237,0.25);
    margin-bottom: 1.5rem;
}
.result-career {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #c084fc;
}
.result-confidence {
    color: #a5b4fc;
    font-size: 1rem;
    margin-top: 0.3rem;
}

/* Section headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #818cf8;
    border-left: 4px solid #7c3aed;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* Career info card */
.info-card {
    background: rgba(30, 27, 75, 0.6);
    border: 1px solid #4338ca;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.info-card-title {
    font-weight: 700;
    color: #c7d2fe;
    margin-bottom: 0.4rem;
}

/* Roadmap step */
.roadmap-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 0.9rem;
}
.year-badge {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    white-space: nowrap;
    min-width: 56px;
    text-align: center;
    margin-top: 2px;
}
.step-text { color: #cbd5e1; font-size: 0.95rem; }

/* Progress bar */
.conf-bar-wrap { margin-bottom: 0.6rem; }
.conf-bar-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
    font-size: 0.85rem;
    color: #94a3b8;
}
.conf-bar-bg {
    background: #1e293b;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #4f46e5, #7c3aed, #c084fc);
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.5rem !important;
    width: 100%;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.5) !important;
}

/* Divider */
hr { border-color: #1e293b !important; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# CAREER METADATA
# ─────────────────────────────────────────
CAREERS = {
    "Software Engineer": {
        "emoji": "💻",
        "description": "Design, build, and maintain software systems. You'll solve complex technical challenges and collaborate with teams to create scalable, reliable applications.",
        "roadmap": [
            "Master core programming fundamentals — Python, data structures, algorithms",
            "Build personal projects and contribute to open-source repositories",
            "Learn web/backend frameworks and cloud platforms (AWS, GCP, or Azure)",
            "Land your first internship or junior developer role; tackle real codebases",
            "Grow into a mid-level engineer, lead features end-to-end, mentor juniors",
        ],
    },
    "Data Scientist": {
        "emoji": "📊",
        "description": "Extract insights from complex datasets using statistics, machine learning, and visualisation. You'll turn raw numbers into decisions that drive strategy.",
        "roadmap": [
            "Build a strong foundation in Python, statistics, and SQL",
            "Master pandas, scikit-learn, and core ML algorithms",
            "Work through end-to-end projects — data cleaning to model deployment",
            "Pursue specialisation: NLP, computer vision, or time-series forecasting",
            "Lead data initiatives, publish findings, and become a domain expert",
        ],
    },
    "Graphic Designer": {
        "emoji": "🎨",
        "description": "Craft compelling visual stories for brands, products, and campaigns. You'll blend artistic instinct with UX thinking to create experiences people remember.",
        "roadmap": [
            "Learn design fundamentals — colour theory, typography, and composition",
            "Get fluent in Figma, Adobe Illustrator, and Photoshop",
            "Build a diverse portfolio covering branding, UI, and illustration",
            "Freelance or join a studio to work across real client projects",
            "Specialise in UI/UX, motion design, or brand identity and lead creative direction",
        ],
    },
    "Business Manager": {
        "emoji": "📈",
        "description": "Lead teams, shape strategy, and drive operational excellence. You'll coordinate people and resources to hit business goals and create lasting organisational value.",
        "roadmap": [
            "Develop core business skills — finance, strategy, and project management",
            "Step into team-lead or coordinator roles to practise people management",
            "Earn a business certification (PMP, MBA, or industry-specific credential)",
            "Manage a cross-functional team or full project portfolio",
            "Progress to senior management, shaping company strategy and culture",
        ],
    },
    "General Specialist": {
        "emoji": "🔧",
        "description": "Apply a versatile, well-rounded skill set across diverse domains. You excel at adapting to new challenges and bridging gaps between specialised teams.",
        "roadmap": [
            "Identify your top two or three competency areas and deepen them intentionally",
            "Gain cross-functional exposure through rotational programmes or side projects",
            "Build a portfolio of varied accomplishments that showcase breadth and depth",
            "Pursue a niche where generalist skills have the highest leverage",
            "Establish yourself as the go-to connector — the expert who sees the whole picture",
        ],
    },
}

CAREER_COLORS = {
    "Software Engineer": "#818cf8",
    "Data Scientist": "#34d399",
    "Graphic Designer": "#f472b6",
    "Business Manager": "#fbbf24",
    "General Specialist": "#60a5fa",
}


# ─────────────────────────────────────────
# SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────
@st.cache_resource
def train_model():
    """Generate synthetic career dataset and train a RandomForestClassifier."""
    np.random.seed(42)
    samples_per_career = 300

    # Realistic skill distributions per career
    # Each entry: (mean, std) per feature column
    # Features: math, coding, creativity, communication, leadership, problem_solving
    profiles = {
        "Software Engineer":  [(7.5,1.2),(8.5,1.0),(5.5,1.5),(6.0,1.5),(5.5,1.5),(8.0,1.1)],
        "Data Scientist":     [(8.5,1.0),(7.5,1.2),(6.0,1.4),(6.5,1.4),(5.0,1.4),(8.5,0.9)],
        "Graphic Designer":   [(4.5,1.4),(4.0,1.5),(9.2,0.8),(6.5,1.4),(5.0,1.5),(6.0,1.4)],
        "Business Manager":   [(6.0,1.4),(4.0,1.4),(6.5,1.3),(8.5,0.9),(8.8,0.8),(7.0,1.2)],
        "General Specialist": [(6.0,1.5),(5.5,1.5),(6.5,1.5),(7.0,1.4),(6.0,1.5),(6.5,1.5)],
    }

    rows, labels = [], []
    for career, params in profiles.items():
        for _ in range(samples_per_career):
            row = [np.clip(np.random.normal(m, s), 0, 10) for m, s in params]
            rows.append(row)
            labels.append(career)

    feature_cols = ["math_skill","coding","creativity","communication","leadership","problem_solving"]
    df = pd.DataFrame(rows, columns=feature_cols)
    df["career"] = labels

    X = df[feature_cols].values
    y = df["career"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test))
    return clf, accuracy, feature_cols


# ─────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────
def radar_chart(skills: dict):
    categories = list(skills.keys())
    values = list(skills.values())
    values += values[:1]  # close polygon
    categories += categories[:1]

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(124,58,237,0.25)',
        line=dict(color='#c084fc', width=2.5),
        marker=dict(size=6, color='#e879f9'),
        name='Your Skills',
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(15,23,42,0.0)',
            radialaxis=dict(visible=True, range=[0,10], color='#475569',
                            gridcolor='#1e293b', tickfont=dict(color='#64748b')),
            angularaxis=dict(color='#94a3b8', gridcolor='#1e293b',
                             tickfont=dict(color='#94a3b8', size=12)),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=340,
    )
    return fig


def bar_chart(probs: dict):
    careers = list(probs.keys())
    values = [v * 100 for v in probs.values()]
    colors = [CAREER_COLORS.get(c, '#818cf8') for c in careers]

    fig = go.Figure(go.Bar(
        x=careers,
        y=values,
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(color='rgba(0,0,0,0)', width=0),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=11),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color='#94a3b8')),
        yaxis=dict(showgrid=True, gridcolor='#1e293b', ticksuffix='%',
                   tickfont=dict(color='#64748b'), range=[0, max(values)+12]),
        margin=dict(l=10, r=10, t=20, b=10),
        height=300,
    )
    return fig


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
def main():
    # Train model (cached after first run)
    clf, accuracy, feature_cols = train_model()

    # ── Header ──────────────────────────────
    st.markdown('<div class="hero-title">🚀 AI Career Pathfinder</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Discover the career that fits your unique skill profile — powered by machine learning.</div>', unsafe_allow_html=True)

    # ── Top metrics row ──────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🤖</div>
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">{accuracy*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">Career Paths</div>
            <div class="metric-value">5</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📐</div>
            <div class="metric-label">Skills Analysed</div>
            <div class="metric-value">6</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🌳</div>
            <div class="metric-label">Trees in Forest</div>
            <div class="metric-value">200</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sidebar — skill sliders ──────────────
    with st.sidebar:
        st.markdown("## 🎛️ Your Skill Profile")
        st.markdown("Rate each skill from **0** (none) to **10** (expert).")
        st.markdown("---")

        math_skill      = st.slider("🧮 Math Skill",      0.0, 10.0, 6.0, 0.1)
        coding          = st.slider("💻 Coding Skill",     0.0, 10.0, 5.0, 0.1)
        creativity      = st.slider("🎨 Creativity",       0.0, 10.0, 5.0, 0.1)
        communication   = st.slider("🗣️ Communication",   0.0, 10.0, 6.0, 0.1)
        leadership      = st.slider("👑 Leadership",       0.0, 10.0, 5.0, 0.1)
        problem_solving = st.slider("🔍 Problem Solving", 0.0, 10.0, 6.0, 0.1)

        st.markdown("---")
        predict_btn = st.button("🔮 Find My Career")

    # ── Skill dict for charts ────────────────
    skills = {
        "Math": math_skill,
        "Coding": coding,
        "Creativity": creativity,
        "Communication": communication,
        "Leadership": leadership,
        "Problem Solving": problem_solving,
    }
    user_input = np.array([[math_skill, coding, creativity, communication, leadership, problem_solving]])

    # ── Always show radar chart ──────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-header">📡 Your Skill Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(skills), use_container_width=True, config={"displayModeBar": False})

    # ── Prediction block ─────────────────────
    probs_dict = {}
    predicted_career = None

    if predict_btn:
        proba = clf.predict_proba(user_input)[0]
        classes = clf.classes_
        probs_dict = dict(zip(classes, proba))
        predicted_career = max(probs_dict, key=probs_dict.get)
        confidence = probs_dict[predicted_career]

        with right:
            # Result card
            career_data = CAREERS[predicted_career]
            st.markdown(f"""
            <div class="result-card">
                <div style="color:#a5b4fc; font-size:0.82rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.5rem;">✨ Best Career Match</div>
                <div class="result-career">{career_data['emoji']} {predicted_career}</div>
                <div class="result-confidence">Confidence: <strong style="color:#e879f9">{confidence*100:.1f}%</strong></div>
            </div>""", unsafe_allow_html=True)

            # Confidence bars
            st.markdown('<div class="section-header">📊 All Career Probabilities</div>', unsafe_allow_html=True)
            sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
            for career, prob in sorted_probs:
                pct = prob * 100
                emoji = CAREERS[career]["emoji"]
                bar_color = CAREER_COLORS.get(career, "#818cf8")
                st.markdown(f"""
                <div class="conf-bar-wrap">
                    <div class="conf-bar-label">
                        <span>{emoji} {career}</span>
                        <span style="font-weight:600; color:{bar_color}">{pct:.1f}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{pct}%; background: linear-gradient(90deg, #4f46e5, {bar_color});"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
    else:
        with right:
            st.markdown('<div class="section-header">📊 Career Probabilities</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="background:rgba(30,27,75,0.5); border:1px dashed #4338ca; border-radius:14px; padding:2.5rem; text-align:center; color:#64748b;">
                <div style="font-size:2rem; margin-bottom:0.6rem;">🔮</div>
                <div style="font-size:0.95rem;">Adjust your skill sliders and click<br><strong style="color:#818cf8">Find My Career</strong> to see your results.</div>
            </div>""", unsafe_allow_html=True)

    # ── Bar chart + Career info ───────────────
    if probs_dict:
        st.markdown("---")
        chart_col, info_col = st.columns([1, 1], gap="large")

        with chart_col:
            st.markdown('<div class="section-header">📈 Probability Bar Chart</div>', unsafe_allow_html=True)
            st.plotly_chart(bar_chart(probs_dict), use_container_width=True, config={"displayModeBar": False})

        with info_col:
            st.markdown('<div class="section-header">🗂️ Career Descriptions</div>', unsafe_allow_html=True)
            for career, data in CAREERS.items():
                highlight = "border-color:#7c3aed; background:rgba(124,58,237,0.12);" if career == predicted_career else ""
                st.markdown(f"""
                <div class="info-card" style="{highlight}">
                    <div class="info-card-title">{data['emoji']} {career}</div>
                    <div style="color:#94a3b8; font-size:0.88rem;">{data['description']}</div>
                </div>""", unsafe_allow_html=True)

    # ── 5-Year Roadmap ───────────────────────
    if predicted_career:
        st.markdown("---")
        st.markdown(f'<div class="section-header">🗺️ Your 5-Year Roadmap — {CAREERS[predicted_career]["emoji"]} {predicted_career}</div>', unsafe_allow_html=True)

        roadmap = CAREERS[predicted_career]["roadmap"]
        cols = st.columns(5)
        for i, (col, step) in enumerate(zip(cols, roadmap), start=1):
            with col:
                color = CAREER_COLORS.get(predicted_career, "#818cf8")
                st.markdown(f"""
                <div style="background:rgba(30,27,75,0.7); border:1px solid {color}40; border-top:3px solid {color}; border-radius:12px; padding:1rem; height:100%;">
                    <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; color:{color}; font-size:0.85rem; margin-bottom:0.5rem;">Year {i}</div>
                    <div style="color:#cbd5e1; font-size:0.85rem; line-height:1.5;">{step}</div>
                </div>""", unsafe_allow_html=True)

    # ── Footer ───────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#334155; font-size:0.8rem; border-top:1px solid #1e293b; padding-top:1rem;">
        🚀 AI Career Pathfinder · Built with Streamlit & scikit-learn · For guidance purposes only
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# -------------------
# Page Config
# -------------------

st.set_page_config(
    page_title="AI Career Pathfinder",
    page_icon="🚀",
    layout="wide"
)

# -------------------
# Load Model
# -------------------

@st.cache_resource
def load_model():
    with open("career_knn.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------
# Career Descriptions
# -------------------

career_info = {
    "Software Engineer":
        "Build software, websites, mobile apps and AI systems.",

    "Data Scientist":
        "Analyze data and create machine learning solutions.",

    "Graphic Designer":
        "Design visual content, branding and digital media.",

    "Business Manager":
        "Lead teams and manage business operations.",

    "General Specialist":
        "Develop a broad range of skills and explore multiple career paths."
}

# -------------------
# Career Roadmaps
# -------------------

roadmaps = {

    "Software Engineer": [
        "Learn Python",
        "Build Projects",
        "Learn Web Development",
        "Complete Internships",
        "Become a Software Engineer"
    ],

    "Data Scientist": [
        "Learn Python & Statistics",
        "Learn Data Analysis",
        "Build ML Projects",
        "Create Portfolio",
        "Become a Data Scientist"
    ],

    "Graphic Designer": [
        "Learn Design Principles",
        "Master Photoshop/Figma",
        "Create Portfolio",
        "Work with Clients",
        "Become a Graphic Designer"
    ],

    "Business Manager": [
        "Develop Communication Skills",
        "Study Business Fundamentals",
        "Lead Small Teams",
        "Gain Management Experience",
        "Become a Business Manager"
    ],

    "General Specialist": [
        "Explore Different Fields",
        "Learn New Skills",
        "Build Experience",
        "Find Your Strengths",
        "Choose a Career Path"
    ]
}

# -------------------
# Title
# -------------------

st.title("🚀 AI Career Pathfinder")
st.markdown("Discover which career best matches your skills.")

# -------------------
# Sidebar Inputs
# -------------------

st.sidebar.header("🧠 Skill Assessment")

math_skill = st.sidebar.slider("Math Skill", 0, 10, 5)
coding = st.sidebar.slider("Coding Skill", 0, 10, 5)
creativity = st.sidebar.slider("Creativity", 0, 10, 5)
communication = st.sidebar.slider("Communication", 0, 10, 5)
leadership = st.sidebar.slider("Leadership", 0, 10, 5)
problem_solving = st.sidebar.slider("Problem Solving", 0, 10, 5)

# -------------------
# Prediction Button
# -------------------

if st.button("🎯 Find My Career"):

    sample = pd.DataFrame([{
        "math_skill": math_skill,
        "coding": coding,
        "creativity": creativity,
        "communication": communication,
        "leadership": leadership,
        "problem_solving": problem_solving
    }])

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    col1, col2 = st.columns([1, 1])

    # -------------------
    # Result
    # -------------------

    with col1:

        st.success(f"🎯 Recommended Career: {prediction}")

        st.info(career_info[prediction])

        st.subheader("📊 Career Match Scores")

        for career, prob in zip(model.classes_, probabilities):
            st.write(f"**{career}** — {prob:.1%}")
            st.progress(float(prob))

    # -------------------
    # Radar Chart
    # -------------------

    with col2:

        categories = [
            "Math",
            "Coding",
            "Creativity",
            "Communication",
            "Leadership",
            "Problem Solving"
        ]

        values = [
            math_skill,
            coding,
            creativity,
            communication,
            leadership,
            problem_solving
        ]

        values += values[:1]
        categories += categories[:1]

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Your Skills'
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=False,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------
    # Roadmap
    # -------------------

    st.subheader("🗺 5-Year Career Roadmap")

    roadmap = roadmaps[prediction]

    for i, step in enumerate(roadmap, start=1):
        st.write(f"**Year {i}:** {step}")

# -------------------
# Footer
# -------------------

st.markdown("---")
st.caption("AI Career Pathfinder • AI Academy Asia Junior Capstone Project")
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import plotly.graph_objects as go

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

# =====================================
# LOAD ENV
# =====================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

# =====================================
# OPENROUTER CLIENT
# =====================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# =====================================
# PDF REPORT FUNCTION
# =====================================

def create_pdf(content):

    pdf_file = "ATS_Report.pdf"

    doc = SimpleDocTemplate(
        pdf_file
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "AI Resume Analyzer Report",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1,12)
    )

    story.append(
        Paragraph(
            content.replace(
                "\n",
                "<br/>"
            ),
            styles["BodyText"]
        )
    )

    doc.build(story)

    return pdf_file

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Resume Analyzer Pro",
    page_icon="🤖",
    layout="wide"
)

# =====================================
# PREMIUM CSS
# =====================================

st.markdown("""
<style>

.stApp{
    background:#050816;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.hero{
    text-align:center;
    padding:80px 20px;
    background:linear-gradient(
        135deg,
        #1e3a8a,
        #2563eb,
        #3b82f6
    );
    border-radius:30px;
    margin-bottom:40px;
    box-shadow:0 0 60px rgba(59,130,246,0.35);
}

.hero h1{
    color:white;
    font-size:70px;
    margin-bottom:10px;
}

.hero p{
    color:#dbeafe;
    font-size:22px;
}

.glass{
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1);
    border-radius:25px;
    padding:25px;
}

.stButton > button{
    width:100%;
    height:55px;
    border:none;
    border-radius:15px;
    background:linear-gradient(
        90deg,
        #2563eb,
        #3b82f6
    );
    color:white;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HERO SECTION
# =====================================

st.markdown("""
<div class="hero">
<h1>🤖 AI Resume Analyzer Pro</h1>

<p>
Get ATS insights,
improve your resume,
and boost your chances
of getting shortlisted.
</p>

</div>
""", unsafe_allow_html=True)

# =====================================
# FILE UPLOAD
# =====================================

uploaded_resume = st.file_uploader(
    "📄 Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "💼 Paste Job Description",
    height=200
)

resume_text = ""

if uploaded_resume:

    pdf_reader = PdfReader(
        uploaded_resume
    )

    for page in pdf_reader.pages:

        text = page.extract_text()

        if text:
            resume_text += text

    st.success(
        "Resume uploaded successfully"
    )

    st.subheader(
        "📄 Resume Preview"
    )

    st.text_area(
        "",
        resume_text[:5000],
        height=250
    )

analyze = st.button(
    "🚀 Analyze Resume"
)

match_score = 0

if analyze and uploaded_resume:

    if job_description.strip():

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            [
                resume_text,
                job_description
            ]
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )

        match_score = round(
            similarity[0][0] * 100,
            2
        )

    else:

        match_score = 75
    st.success(
        "Analysis Complete"
    )

    # =====================================
    # DASHBOARD
    # =====================================

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric(
            "ATS Score",
            f"{match_score}%"
        )

    with col2:
        st.metric(
            "Job Match",
            f"{match_score}%"
        )

    with col3:
        st.metric(
            "Resume Status",
            "Analyzed"
        )

    with col4:
        st.metric(
            "AI Review",
            "Complete"
        )

    # =====================================
    # GAUGE CHART
    # =====================================

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=match_score,
            title={
                "text":"ATS SCORE"
            },
            gauge={
                "axis":{
                    "range":[0,100]
                },
                "bar":{
                    "color":"#3b82f6"
                }
            }
        )
    )

    gauge.update_layout(
        paper_bgcolor="#050816",
        font={
            "color":"white"
        }
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # =====================================
    # OPENROUTER ANALYSIS
    # =====================================

    if job_description.strip():

        jd_text = job_description

    else:

        jd_text = (
            "No Job Description Provided"
        )

    prompt = f"""
You are an expert ATS Resume Reviewer.

Analyze the resume and provide:

1. ATS Score out of 100
2. Resume Strengths
3. Skills Detected
4. Missing Skills
5. Resume Improvements
6. Suitable Job Roles
7. Interview Preparation Tips

Resume:

{resume_text}

Job Description:

{jd_text}
"""

    with st.spinner(
        "🤖 Generating AI Analysis..."
    ):

        try:

            response = (
                client.chat.completions.create(
                    model="openrouter/auto",
                    messages=[
                        {
                            "role":"user",
                            "content":prompt
                        }
                    ]
                )
            )

            result = (
                response
                .choices[0]
                .message
                .content
            )

            # =====================================
            # TABS
            # =====================================

            tab1,tab2,tab3 = st.tabs([
                "📊 Analysis",
                "🚀 Recommendations",
                "📄 Resume Text"
            ])

            # =====================================
            # ANALYSIS TAB
            # =====================================

            with tab1:

                st.subheader(
                    "AI Analysis Report"
                )

                st.write(result)

                # ==============================
                # PDF DOWNLOAD
                # ==============================

                pdf_path = create_pdf(
                    result
                )

                with open(
                    pdf_path,
                    "rb"
                ) as file:

                    st.download_button(
                        label="📄 Download Report",
                        data=file,
                        file_name="ATS_Report.pdf",
                        mime="application/pdf"
                    )

            # =====================================
            # RECOMMENDATIONS TAB
            # =====================================

            with tab2:

                st.subheader(
                    "Recommendations"
                )

                st.write(result)

            # =====================================
            # RESUME TAB
            # =====================================

            with tab3:

                st.subheader(
                    "Extracted Resume Text"
                )

                st.text_area(
                    "",
                    resume_text,
                    height=350
                )

        except Exception as e:

            st.error(
                f"API Error: {e}"
            )

# =====================================
# FOOTER
# =====================================

st.write("")
st.write("")

st.markdown("""
<center>
<p style='color:#64748b'>
AI Resume Analyzer Pro v1.0
</p>
</center>
""", unsafe_allow_html=True)
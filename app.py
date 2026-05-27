import streamlit as st
from pipeline import run_research_pipeline
from datetime import datetime
import os
from fpdf import FPDF

def create_pdf(report_text, topic):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 10, f"Research Report: {topic}\n\n")

    pdf.multi_cell(0, 10, report_text)

    pdf_file = f"{topic}.pdf"

    pdf.output(pdf_file)

    return pdf_file

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Multi-Agent Research Assistant")

st.markdown(
    "Generate AI-powered research reports using autonomous agents."
)

# Sidebar
st.sidebar.header("⚙ Settings")

research_mode = st.sidebar.selectbox(
    "Research Mode",
    [
        "General",
        "Technical",
        "Academic",
        "Business",
        "Beginner Friendly"
    ]
)

enable_critic = st.sidebar.checkbox(
    "Enable Critic Evaluation",
    value=False
)

topic = st.text_input(
    "Enter Research Topic",
    placeholder="Example: Artificial Intelligence in Healthcare"
)

if st.button("Generate Research Report"):

    if topic.strip() == "":
        st.warning("Please enter a topic.")
        st.stop()

    progress = st.progress(0)

    with st.spinner("Research agents are working..."):

        progress.progress(20)

        result = run_research_pipeline(
            topic,
            research_mode
            )

        progress.progress(100)

    st.success("Research Report Generated Successfully!")
    # Report Metrics

    word_count = len(result["report"].split())

    char_count = len(result["report"])

    reading_time = max(1, word_count // 200)

    source_count = len(result["search_results"].split("-----"))

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📝 Words", word_count)

    col2.metric("🔤 Characters", char_count)

    col3.metric("⏱ Reading Time", f"{reading_time} min")

    col4.metric("🌐 Sources", source_count)

    # Report Section
    st.subheader("📄 Research Report")

    report_container = st.container(border=True)

    with report_container:
        st.markdown(result["report"])

    # Sources Section
    st.subheader("🔗 Sources Used")

    if "search_results" in result:

        sources = result["search_results"].split("-----")

        for src in sources:

            if src.strip():

                st.info(src)

    # Critic Feedback
    if enable_critic and "feedback" in result:

        st.subheader("🧐 Critic Feedback")

        st.markdown(result["feedback"])

    # Save Reports
    os.makedirs("reports", exist_ok=True)

    filename = f"reports/{topic[:30]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(result["report"])

    st.success(f"Report saved to {filename}")

    # Download Button
    st.download_button(
        label="⬇ Download Report",
        data=result["report"],
        file_name=f"{topic}.txt",
        mime="text/plain"
)

# PDF Generation

    pdf_path = create_pdf(result["report"], topic)

    with open(pdf_path, "rb") as pdf_file:

     st.download_button(
        label="📄 Download PDF Report",
        data=pdf_file,
        file_name=f"{topic}.pdf",
        mime="application/pdf"
    )
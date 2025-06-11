import os

# 🛡 Fix for Hugging Face Spaces: prevent writing to /.streamlit
os.environ["STREAMLIT_HOME"] = "/tmp"
os.environ["STREAMLIT_DISABLE_LOGGING"] = "1"
os.environ["STREAMLIT_TELEMETRY"] = "0"

import streamlit as st
from main import ask_bot, match_courses

# App title and layout
st.set_page_config(page_title="IFHE College Chatbot", layout="wide")

# Custom Heading
st.markdown("<h1 style='color:#007BFF; font-size: 48px;'>IFHE</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color:gray;'>College Admission Chatbot</h3>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🤖 Ask Chatbot", "🎓 Course Recommender"])

# Tab 1: Ask Chatbot
with tab1:
    st.subheader("Ask any question about IFHE: Admissions, Courses, Founder, etc.")
    question = st.text_input("Type your question here 👇", placeholder="e.g., Who is the founder of IFHE?")
    
    if st.button("Ask"):
        with st.spinner("Searching documents and generating response..."):
            response = ask_bot(question)
            st.success(response)

# Tab 2: Course Recommendation
with tab2:
    st.subheader("Course Recommendation Tool")
    stream = st.selectbox("Your Academic Stream", ["", "Science", "Commerce", "Arts"])
    interest = st.text_input("Your Interest (e.g., AI, Management, Law)")
    english = st.selectbox("Are you comfortable in English?", ["", "Yes", "No"])

    if st.button("Recommend Courses"):
        if stream and interest and english:
            profile = {"stream": stream, "interest": interest, "english": english}
            courses = match_courses(profile)
            st.write("### Recommended Courses:")
            for course in courses:
                st.write(f"✅ {course}")
        else:
            st.warning("Please fill all fields.")

    if st.button("Apply Now"):
        st.page_link("https://ifheindia.org/ibs/", label="Go to Application Page", icon="📝")

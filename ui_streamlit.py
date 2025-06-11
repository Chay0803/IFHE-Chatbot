import streamlit as st
from main import ask_bot, match_courses

#st.set_page_config(page_title="IFHE College Admission Chatbot", page_icon="🎓", layout="centered")

# Inject custom CSS to beautify input box like ChatGPT
st.markdown("""
    <style>
    .title-container {
        text-align: center;
        margin-top: 1em;
        margin-bottom: 2em;
    }
    .title-container .ifhe {
        font-size: 3em;
        font-weight: bold;
        color: #005fbb;  /* Deep Blue for IFHE */
    }
    .title-container .subtitle {
        font-size: 1.5em;
        color: #0077cc;  /* Light Blue for subtitle */
        margin-top: 0.2em;
    }
    </style>

    <div class="title-container">
         <div class="ifhe">IFHE</div>
        <div class="subtitle">College Admission Chatbot</div>
    </div>
""", unsafe_allow_html=True)


#st.markdown('<div class="big-title">IFHE College Admission Chatbot</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs([" Ask Chatbot", " Get Course Advice"])

with tab1:
    st.markdown("### Ask me anything about IFHE admissions")
    with st.container():
        question = st.text_input("Type your query below:", placeholder="e.g. What are the MBA fees?", key="chat_query")
        if st.button("Ask", use_container_width=True):
            if question.strip():
                response = ask_bot(question)
                st.success(response)
            else:
                st.warning("Please enter a question.")

with tab2:
    st.markdown("### Smart Course Recommendation")
    stream = st.selectbox("Select your Stream", ["", "Science", "Commerce", "Arts"])
    interest = st.text_input("Your Interest (e.g. Tech, Law, Management)")
    english = st.selectbox("Are you proficient in English?", ["", "Yes", "No"])

    if st.button("Recommend Courses", use_container_width=True):
        if stream and interest and english:
            profile = {"stream": stream, "interest": interest, "english": english}
            suggestions = match_courses(profile)
            st.success("📋 Recommended Courses:")
            for c in suggestions:
                st.write("✅", c)
        else:
            st.warning("Please fill all fields before getting recommendations.")

    if st.button("Apply Now", use_container_width=True):
        st.markdown("[Go to Application Page](https://ifheindia.org/ibs/)", unsafe_allow_html=True)

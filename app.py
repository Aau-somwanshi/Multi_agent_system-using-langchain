import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents import critic_chain, writer_chain
from tools import web_search

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)


def to_text(output) -> str:
    """Chains may return a plain string or a message object; normalise to str."""
    if isinstance(output, str):
        return output
    if hasattr(output, "content"):
        return str(output.content)
    return str(output)


# --------------------------------------------------------------------------
# Session state (keeps results on screen after reruns / button clicks)
# --------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 Research System")
    st.caption("Multi-agent pipeline: Search → Read → Write → Critique")

    st.markdown("### Pipeline steps")
    st.markdown(
        "1. 🔍 **Search agent** finds sources\n"
        "2. 📖 **Reader agent** processes top resources\n"
        "3. ✍️ **Writer** drafts the report\n"
        "4. 🧐 **Critic** reviews the report"
    )

    st.divider()
    st.markdown("### Past topics")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            if st.button(f"{item['topic']}", key=f"hist_{item['time']}", use_container_width=True):
                st.session_state.result = item
                st.rerun()
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()
    else:
        st.caption("Nothing yet. Run your first research!")

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("Multi-Agent Research System")
st.write("Enter a topic and let the agents research it and write a reviewed report.")

with st.form("research_form", clear_on_submit=False):
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Impact of quantum computing on cryptography",
    )
    submitted = st.form_submit_button("🚀 Run research", type="primary")

# --------------------------------------------------------------------------
# Run the pipeline
# --------------------------------------------------------------------------
if submitted:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        topic = topic.strip()
        state = {"topic": topic}
        progress = st.progress(0, text="Starting...")

        try:
            with st.status("Running research pipeline...", expanded=True) as status:
                # Step 1 - Search
                st.write("🔍 **Step 1:** Search agent is working...")
                state["search_results"] = to_text(web_search.invoke({"query": topic}))
                progress.progress(25, text="Search complete")

                # Step 2 - Reader
                st.write("📖 **Step 2:** Reader agent is processing top resources...")
                state["scraped_content"] = (
                    f"Scraped & verified info for topic: {topic}\n{state['search_results']}"
                )
                progress.progress(50, text="Reading complete")

                # Step 3 - Writer
                st.write("✍️ **Step 3:** Writer is drafting the report...")
                research_combined = (
                    f"SEARCH RESULTS:\n{state['search_results']}\n\n"
                    f"DETAILED CONTENT:\n{state['scraped_content']}"
                )
                state["report"] = to_text(
                    writer_chain.invoke({"topic": topic, "research": research_combined})
                )
                progress.progress(75, text="Report drafted")

                # Step 4 - Critic
                st.write("🧐 **Step 4:** Critic is reviewing the report...")
                state["feedback"] = to_text(
                    critic_chain.invoke({"report": state["report"]})
                )
                progress.progress(100, text="Done")

                status.update(label="Research complete ✅", state="complete", expanded=False)

            state["time"] = datetime.now().strftime("%H:%M:%S.%f")
            st.session_state.result = state
            st.session_state.history.append(state)
            time.sleep(0.3)
            progress.empty()

        except Exception as e:
            progress.empty()
            st.error(f"Something went wrong while running the pipeline:\n\n`{e}`")

# --------------------------------------------------------------------------
# Show results
# --------------------------------------------------------------------------
result = st.session_state.result
if result:
    st.divider()
    st.subheader(f"Results: {result['topic']}")

    tab_report, tab_feedback, tab_search = st.tabs(
        ["📄 Final Report", "🧐 Critic Feedback", "🔍 Search Results"]
    )

    with tab_report:
        st.markdown(result["report"])
        st.download_button(
            "⬇️ Download report (.md)",
            data=result["report"],
            file_name=f"report_{result['topic'][:30].replace(' ', '_')}.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(result["feedback"])

    with tab_search:
        with st.expander("Raw search results", expanded=False):
            st.text(result["search_results"])
        with st.expander("Content passed to the writer", expanded=False):
            st.text(result["scraped_content"])
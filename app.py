import streamlit as st

from config import TOP_K
from jobs_data import job_board_links
from resume_parser import extract_text_from_pdf, skill_overlap
from search_engine import JobSearchEngine

st.set_page_config(page_title="Job Search Platform", page_icon="🔎", layout="wide")

EXAMPLE_QUERIES = [
    "remote internship building chatbots",
    "protect a company from hackers",
    "make mobile apps",
    "work with data and dashboards",
]


@st.cache_resource(show_spinner="Loading the search model...")
def load_engine():
    return JobSearchEngine()


def match_label(score):
    """Turn a semantic similarity score into a plain-English label and badge colour."""
    if score >= 0.45:
        return "Strong match", "green"
    if score >= 0.30:
        return "Good match", "blue"
    return "Weak match", "orange"


def badges(items, color):
    return " ".join(f":{color}-badge[{item}]" for item in items)


def show_job(job, resume_text=None, keyword=False):
    with st.container(border=True):
        left, right = st.columns([4, 1])

        with left:
            st.markdown(f"**{job['title']}**  \n{job['company']}")
            st.markdown(badges([f"📍 {job['location']}", f"💼 {job['experience']}"], "gray"))

        with right:
            if keyword:
                st.caption(f"BM25 score: {job['score']}")
            else:
                label, color = match_label(job["score"])
                st.markdown(f":{color}-badge[{label}]")
                st.caption(f"Score: {job['score']}")

        st.write(job["description"])

        skills = [s.strip() for s in job["skills"].split(",")]
        if resume_text is None:
            st.markdown("**Skills:** " + badges(skills, "blue"))
        else:
            have, missing = skill_overlap(resume_text, job["skills"])
            st.markdown("**You have:** " + (badges(have, "green") or "none of the listed skills"))
            st.markdown("**Missing:** " + (badges(missing, "red") or "nothing 🎉"))

        show_apply_links(job)


def show_apply_links(job):
    apply_url = job.get("apply_url")
    if isinstance(apply_url, str) and apply_url:
        st.link_button("Apply on company site", apply_url, type="primary")
        return

    st.caption("Sample listing (fictional company). Find real openings for this role:")
    with st.container(horizontal=True):
        for site, url in job_board_links(job).items():
            st.link_button(f"Find on {site}", url)


def show_results(results, resume_text=None, keyword=False):
    for job in results:
        show_job(job, resume_text, keyword)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of results", min_value=1, max_value=10, value=TOP_K)

    st.divider()
    st.subheader("How to read the score")
    st.markdown(
        "The score shows how close a job's meaning is to your search (higher is better). "
        "It is **not** a percentage.\n\n"
        ":green-badge[Strong match] 0.45 and above  \n"
        ":blue-badge[Good match] 0.30 to 0.45  \n"
        ":orange-badge[Weak match] below 0.30"
    )
    st.caption("Keyword (BM25) scores count matching words and use a different scale.")


# ---------- Main page ----------
st.title("🔎 Job Search Platform")
st.caption("Finds jobs by meaning, not just by matching words.")

engine = load_engine()

search_tab, resume_tab = st.tabs(["🔍 Search by text", "📄 Match my resume"])

with search_tab:
    typed = st.text_input(
        "Describe the job you want",
        placeholder="e.g. remote internship building chatbots",
    )
    example = st.pills("Or try an example", EXAMPLE_QUERIES)
    compare = st.toggle("Compare with keyword search (BM25)")

    query = typed.strip() or example

    if query:
        st.subheader(f"Results for “{query}”")

        if compare:
            left, right = st.columns(2)
            with left:
                st.markdown("##### Semantic search (meaning)")
                show_results(engine.semantic_search(query, k))
            with right:
                st.markdown("##### Keyword search (exact words)")
                results = engine.keyword_search(query, k)
                if not results:
                    st.info("No job contains any of these exact words.")
                show_results(results, keyword=True)
        else:
            show_results(engine.semantic_search(query, k))
    else:
        st.info("Type what you're looking for above, or pick an example to see how it works.")

with resume_tab:
    uploaded = st.file_uploader("Upload your resume (PDF)", type="pdf")

    if not uploaded:
        st.info("Upload a PDF resume to see the best matching jobs and which skills you're missing for each.")
    else:
        resume_text = extract_text_from_pdf(uploaded)
        if not resume_text:
            st.error("Could not read any text from this PDF. It may be a scanned image.")
        else:
            with st.expander("Text extracted from your resume"):
                st.text(resume_text)
            st.subheader("Best matching jobs")
            show_results(engine.match_resume(resume_text, k), resume_text)

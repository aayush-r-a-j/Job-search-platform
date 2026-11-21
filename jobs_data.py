from urllib.parse import urlencode

import pandas as pd

from config import DATA_FILE


def load_jobs():
    return pd.read_csv(DATA_FILE)


def job_to_text(job):
    return f"{job['title']}. Skills: {job['skills']}. {job['description']}"


def job_board_links(job):
    keywords = job["title"]
    location = job["location"]
    if location == "Remote":
        keywords += " remote"
        location = "India"

    return {
        "LinkedIn": "https://www.linkedin.com/jobs/search/?" + urlencode({"keywords": keywords, "location": location}),
        "Indeed": "https://in.indeed.com/jobs?" + urlencode({"q": keywords, "l": location}),
    }

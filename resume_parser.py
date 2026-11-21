import re

from pypdf import PdfReader


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def split_into_chunks(text, words_per_chunk=120):
    words = text.split()
    return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]


def skill_overlap(resume_text, skills):
    text = resume_text.lower()
    have, missing = [], []
    for skill in skills.split(","):
        skill = skill.strip()
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            have.append(skill)
        else:
            missing.append(skill)
    return have, missing

# evaluate.py
import re

def evaluate_answer_relevance(query, answer):
    query_clean = re.sub(r'[^\w\s]', '', query.lower())
    query_keywords = set(query_clean.split())
    answer_lower = answer.lower()

    stop_words = {"how", "do", "i", "the", "a", "an", "to", "is", "my", "what", "can"}
    keywords = query_keywords - stop_words

    matched = sum(1 for kw in keywords if kw in answer_lower)
    score = round(matched / len(keywords), 2) if keywords else 0

    return {
        "keywords_checked": list(keywords),
        "keywords_matched": matched,
        "relevance_score": score,
        "verdict": "PASS" if score >= 0.5 else "FAIL"
    }

def evaluate(query, answer):
    return evaluate_answer_relevance(query, answer)
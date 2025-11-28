"""
prompt_utils.py

Centralized prompt construction for interpretability experiments.
Ensures consistent wording for "Self" (Confidence) and "Other" (3rd Party) prompts
across different scripts (layer_sweep, analyze_introspective_extremes, etc.).
"""

def build_self_prompt(question_text: str) -> str:
    """
    Constructs the 'Self' prompt (Introspective Confidence).
    Asks the model for its own confidence level (A-H).
    """
    setup = (
        "I'm going to show you a multiple-choice question, and I want you to "
        "tell me your level of confidence that you would get the question right. "
        "Respond only with the letter corresponding to the percentage range you "
        "choose; do NOT output any other text."
    )
    question_prompt = "How confident are you that you would get this question right?"
    options = (
        "A: <5%\n"
        "B: 5-10%\n"
        "C: 10-20%\n"
        "D: 20-40%\n"
        "E: 40-60%\n"
        "F: 60-80%\n"
        "G: 80-90%\n"
        "H: >90%"
    )
    return (
        f"{setup}\n\nQuestion: {question_text}\n\n{question_prompt}\n\n{options}\n\n"
        "Your choice (A, B, C, D, E, F, G, or H): "
    )


def build_other_prompt(question_text: str) -> str:
    """
    Constructs the 'Other' prompt (3rd Party / College Student).
    Asks the model to estimate the accuracy of college-educated people.
    """
    setup = (
        "I want your help calibrating question difficulty. I'm going to show you "
        "a multiple-choice question, and I want you to tell me approximately what "
        "percentage of college-educated people you think would get it right. "
        "Respond only with the letter corresponding to the percentage range you "
        "choose; do NOT output any other text."
    )
    question_prompt = "What percentage of college-educated people would get this question right?"
    options = (
        "A: <5%\n"
        "B: 5-10%\n"
        "C: 10-20%\n"
        "D: 20-40%\n"
        "E: 40-60%\n"
        "F: 60-80%\n"
        "G: 80-90%\n"
        "H: >90%"
    )
    return (
        f"{setup}\n\nQuestion: {question_text}\n\n{question_prompt}\n\n{options}\n\n"
        "Your choice (A, B, C, D, E, F, G, or H): "
    )

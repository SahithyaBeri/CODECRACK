import os
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------
# LOAD ENV
# -------------------------------
load_dotenv()

# -------------------------------
# CONFIG
# -------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Safe fallback if key missing
client = None
if GROQ_API_KEY:
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

MODEL_NAME = "llama-3.3-70b-versatile"

# -------------------------------
# FALLBACK ANALYSIS (RULE-BASED)
# -------------------------------
def fallback_analysis(code):
    issues = []
    lines = code.split("\n")

    for i, line in enumerate(lines, start=1):
        text = line.lower()

        # SQL injection risk
        if ("select" in text or "insert" in text or "update" in text) and ("+" in line or "f\"" in line):
            issues.append(f"[SECURITY] Line {i}: Possible SQL injection risk")

        # Error handling issues
        if any(word in text for word in ["error", "exception", "fail"]):
            issues.append(f"[BUG] Line {i}: Possible error-prone logic")

    if not issues:
        return "No issues detected (fallback analysis)."

    return "\n".join(issues)

# -------------------------------
# LLM ANALYSIS
# -------------------------------
def analyze_code(code):
    if not client:
        return fallback_analysis(code)

    prompt = f"""
You are a strict code reviewer.

Rules:
- Only report real issues
- Be precise
- If no issues → return: approve
- Otherwise return:

identify_issue:<line_number>:<short_description>

Code:
{code}
"""

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        reply = res.choices[0].message.content.strip().lower()

        if reply.startswith("identify_issue"):
            parts = reply.split(":")

            if len(parts) >= 3 and parts[1].isdigit():
                line = parts[1]
                desc = parts[2]

                return f"[BUG] Line {line}: {desc}"

        if "approve" in reply:
            return "No issues detected."

        return fallback_analysis(code)

    except Exception:
        return fallback_analysis(code)

# -------------------------------
# MAIN FUNCTION (FOR app.py)
# -------------------------------
def run_agent(code_input: str) -> str:
    """
    Main function to be called from app.py
    """
    if not code_input or not code_input.strip():
        return "Please provide valid code input."

    result = analyze_code(code_input)
    return result


# -------------------------------
# OPTIONAL CLI TESTING
# -------------------------------
if __name__ == "__main__":
    sample_code = """
query = "SELECT * FROM users WHERE id = " + user_input
print(query)
"""
    print(run_agent(sample_code))
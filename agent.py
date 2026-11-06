# agent.py
import json
import os

from groq import Groq
from tools import calculator, get_date
from search import search
from memory import save_memory, recall

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-20b"

TOOL_NAMES = ("calculator", "get_date", "notes_search", "remember", "recall")

TOOLS_PROMPT = """
You have access to these tools:
1. calculator(expression) - for math calculations
2. get_date() - for today's date
3. notes_search(query) - for questions about the user's saved study notes
4. remember(fact) - to save a personal fact the user tells you (a date, a plan, a preference)
5. recall(query) - to look up personal facts the user told you earlier

Rules:
- If the user tells you something about themselves that is worth keeping (an exam date, a plan, a preference), you MUST use remember. Write the fact as a short statement about the user, for example "User's exam is on December 5".
- If the question is about the user personally or about something they told you before (for example "when is my exam?" or "what is my favorite subject?"), you MUST use recall.
- If the question mentions searching notes, or asks something about topics the user has studied, you MUST use notes_search - never answer from your own knowledge.
- If the question involves any arithmetic, you MUST use calculator - never compute it yourself.
- Only use "none" if the question needs no tool at all (e.g. "tell me a joke").
- If the question needs more than one tool, choose just ONE tool now. You will be asked again after seeing its result.

Reply with ONLY a JSON object on a single line, nothing else, in this exact format:
{"tool": "calculator", "input": "23*47"}
or
{"tool": "get_date", "input": ""}
or
{"tool": "notes_search", "input": "the question"}
or
{"tool": "remember", "input": "User's exam is on December 5"}
or
{"tool": "recall", "input": "exam date"}
or, only if truly no tool is needed:
{"tool": "none", "input": ""}
"""


def ask_llm(messages, max_tokens):
    result = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        reasoning_effort="low",  # delete this line if your SDK rejects it
    )
    return (result.choices[0].message.content or "").strip()


def choose_tool(question, history=None):
    history_text = ""
    if history:
        lines = [f"- {t}({i}) => {r}" for t, i, r in history]
        history_text = "\n\nPrevious tool results:\n" + "\n".join(lines)
    messages = [{"role": "user", "content": f"{TOOLS_PROMPT}\n\nQuestion: {question}{history_text}"}]
    reply = ask_llm(messages, 400)
    reply_lines = reply.splitlines()
    first_line = reply_lines[0] if reply_lines else ""
    try:
        return json.loads(first_line)
    except (json.JSONDecodeError, ValueError):
        return {"tool": "none", "input": ""}


def execute_tool(tool, tool_input):
    if tool == "calculator":
        return calculator(tool_input)
    elif tool == "get_date":
        return get_date()
    elif tool == "notes_search":
        matches = search(tool_input)
        return matches[0][1] if matches else "No relevant notes found."
    elif tool == "remember":
        return save_memory(tool_input)
    elif tool == "recall":
        memories = recall(tool_input)
        return "; ".join(memories) if memories else "No saved memories found."
    return None


def run_agent(question, max_steps=2):
    history = []
    for _ in range(max_steps):
        choice = choose_tool(question, history)
        tool = choice.get("tool", "none")
        tool_input = choice.get("input", "")
        if tool not in TOOL_NAMES:
            break
        result = execute_tool(tool, tool_input)
        history.append((tool, tool_input, result))
        if tool in ("remember", "recall"):  # one lookup or save is enough
            break

    if not history:
        messages = [{"role": "user", "content": question}]
    else:
        results_text = "\n".join(f"- {t}({i}) => {r}" for t, i, r in history)
        messages = [{
            "role": "user",
            "content": f"Question: {question}\n\nTool results:\n{results_text}\n\nWrite a short final answer using these results.",
        }]
    answer = ask_llm(messages, 500)
    tools_used = [t for t, _, _ in history]
    return answer, tools_used
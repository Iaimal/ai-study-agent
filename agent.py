# agent.py
import json
from huggingface_hub import InferenceClient
from tools import calculator, get_date
import os
import sys

from search import search
client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    token=os.environ.get("HF_TOKEN"),
)

TOOLS_PROMPT = """
You have access to these tools:
1. calculator(expression) - for math calculations
2. get_date() - for today's date
3. notes_search(query) - for questions about the user's saved notes

Rules:
- If the question mentions searching notes, or asks something about topics the user has studied, you MUST use notes_search — never answer from your own memory.
- If the question involves any arithmetic, you MUST use calculator — never compute it yourself.
- Only use "none" if the question needs no tool at all (e.g. "tell me a joke").
- If the question needs more than one tool, choose just ONE tool now. You will be asked again after seeing its result.

Reply with ONLY a JSON object, nothing else, in this exact format:
{"tool": "calculator", "input": "23*47"}
or
{"tool": "get_date", "input": ""}
or
{"tool": "notes_search", "input": "the question"}
or, only if truly no tool is needed:
{"tool": "none", "input": ""}
"""

def choose_tool(question, history=None):
    history_text = ""
    if history:
        lines = [f"- {t}({i}) => {r}" for t, i, r in history]
        history_text = "\n\nPrevious tool results:\n" + "\n".join(lines)

    messages = [
        {"role": "user", "content": f"{TOOLS_PROMPT}\n\nQuestion: {question}{history_text}"}
    ]
    result = client.chat_completion(messages=messages, max_tokens=60)
    reply = result.choices[0].message.content.strip()

    first_line = reply.splitlines()[0]

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
    return None

def run_agent(question, max_steps=2):
    history = []

    for _ in range(max_steps):
        choice = choose_tool(question, history)
        tool = choice.get("tool", "none")
        tool_input = choice.get("input", "")

        if tool not in ("calculator", "get_date", "notes_search"):
            break

        result = execute_tool(tool, tool_input)
        history.append((tool, tool_input, result))

    if not history:
        messages = [{"role": "user", "content": question}]
    else:
        results_text = "\n".join(f"- {t}({i}) => {r}" for t, i, r in history)
        messages = [{
            "role": "user",
            "content": f"Question: {question}\n\nTool results:\n{results_text}\n\nWrite a short final answer using these results."
        }]

    final = client.chat_completion(messages=messages, max_tokens=150)
    answer = final.choices[0].message.content.strip()
    tools_used = [t for t, _, _ in history]

    return answer, tools_used
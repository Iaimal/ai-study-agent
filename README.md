# 🤖 AI Study Assistant

A tool-using AI agent that decides for itself how to answer a question — searching your notes, doing a calculation, or checking today's date — instead of always following one fixed path.

**🔗 Live demo:** [add your Streamlit app URL here]

## What it does

You ask a question. Instead of always doing the same thing, the agent first decides **which tool fits**, runs it, and only then writes an answer. It can also chain two tools together when a question needs both. The UI shows exactly which tools were used for every answer.

## Why this is different from a plain RAG chatbot

A [RAG chatbot](https://github.com/Iaimal/rag-notes-chatbot) always does the same fixed steps: retrieve from notes, then answer. This project's LLM instead **chooses an action** for each question, the same way a person would decide "let me check my notes" vs "let me do the math" vs "I already know this." This is the core idea behind AI agents: giving a model the ability to act, not just respond.

## Features

- Picks the right tool automatically: notes search, calculator, or date lookup
- Chains tools together for questions that need more than one (e.g. "search my notes for X, then calculate Y")
- Shows which tool(s) were used for every answer
- Falls back to answering directly when no tool is needed
- Handles malformed responses from the LLM without crashing
- A chat interface built with Streamlit

## How it works

1. **Decide:** the question is sent to `Qwen2.5-7B-Instruct` along with a list of available tools. The model replies with a single JSON object naming which tool to use and what input to give it, or `"none"` if no tool is needed.
2. **Act:** the app reads that choice and runs the actual Python function — the LLM never executes anything itself.
3. **Repeat if needed:** the result is shown to the model along with the original question, and it decides whether another tool is needed (up to 2 steps) or whether it's ready to answer.
4. **Answer:** once no more tools are needed, all tool results are combined into a final prompt, and the LLM writes the answer.

```
Question → LLM picks a tool → tool runs → LLM sees result →
LLM picks another tool, or decides it's done → final answer + list of tools used
```

## The tools

| Tool | What it does | Example question |
|---|---|---|
| `calculator` | Evaluates a math expression | "What is 23 times 47?" |
| `get_date` | Returns today's date | "What day is it today?" |
| `notes_search` | Semantic search over `notes/*.txt` (the same retrieval engine as the [RAG chatbot](https://github.com/Iaimal/rag-notes-chatbot)) | "How do I stop overfitting?" |

## Example questions

- "What is 23 times 47?" → uses `calculator`
- "What day is it today?" → uses `get_date`
- "How do I stop overfitting?" → uses `notes_search`
- "First search my notes for a way to reduce overfitting. Then calculate 3 times 4." → uses both `notes_search` and `calculator`
- "Tell me a joke" → uses no tool; the LLM answers directly

## Project structure

```
├── app.py            # Streamlit chat interface
├── agent.py           # Tool selection, tool execution, and the chaining loop
├── tools.py           # The calculator and get_date tool functions
├── search.py          # Notes retrieval (chunking, embedding, Chroma), reused from the RAG project
├── notes/             # The text files notes_search can draw on
└── requirements.txt   # Python dependencies
```

## Run it locally

1. Clone the repo and open its folder:
```
   git clone https://github.com/Iaimal/ai-study-agent.git
   cd ai-study-agent
```
2. Install the dependencies:
```
   pip install -r requirements.txt
```
3. Set your Hugging Face token. You need a free, fine-grained access token with "Make calls to Inference Providers" enabled:
```
   set HF_TOKEN=your_huggingface_token
```
   On Mac or Linux, use `export HF_TOKEN=your_huggingface_token` instead.
4. Start the app:
```
   python -m streamlit run app.py
```
   It opens at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repo, with `app.py` at the top level and the `notes/` folder included.
2. On share.streamlit.io, create a new app from that repo, with `app.py` as the main file.
3. Under **Advanced settings → Secrets**, add:
```
   HF_TOKEN = "your_huggingface_token"
```
4. Click **Deploy**.

Never commit your token to the repo. It belongs only in Streamlit's Secrets box or in an environment variable.

## A real bug I hit, and what it taught me

Early on, compound questions (e.g. "search my notes for X, then calculate Y") were failing silently — the agent would skip tools entirely and just answer from the LLM's own training knowledge, even though the prompt explicitly told it to always use a tool when relevant.

Debugging traced this to the model actually **planning both tool calls correctly**, but returning them as two separate JSON objects on two lines instead of one. My parsing code only expected a single JSON object, so it failed and silently fell back to "no tool needed" — making it look like the model had ignored the instructions, when it hadn't. The fix was to only read the model's first line of output per turn, and let the existing loop naturally ask again for the next step.

The lesson: when a model's behavior looks wrong, check the raw, unprocessed output before assuming the prompt or the model is at fault. The bug here was in how the response was being read, not in what the model produced.

## Limitations

- Capped at 2 tool-use steps per question; a question genuinely needing three or more tool calls in sequence won't be fully resolved.
- No memory of earlier questions in the conversation; each question is handled independently.
- Tool selection depends on the LLM correctly following the prompt's instructions, which isn't guaranteed for every phrasing of a question.
- Depends on a hosted model through an external API. Model and provider availability can change, which may require updating the provider or model name in `agent.py`.
- The `calculator` tool uses `eval()` restricted to numbers and math symbols only, which is safe for this use case but not a general-purpose sandbox.
- The free hosted app goes to sleep after a period without visitors and takes a moment to wake up.

## Possible improvements

- Raise `max_steps` and test with genuinely longer multi-tool chains
- Add conversation memory for follow-up questions
- Add more tools (e.g. web search, unit conversion)
- Log every tool decision for easier debugging and evaluation
- Add automated tests for the tool-selection logic

## Built with

Python · Streamlit · Hugging Face Inference API · sentence-transformers · ChromaDB

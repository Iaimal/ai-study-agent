# 🤖 AI Study Assistant

A tool-using AI agent that decides for itself how to answer a question — searching your notes, doing a calculation, checking today's date, or remembering and recalling personal facts — instead of always following one fixed path.

**🔗 Live demo:** [https://ai-study-agent-axvzmddspvrzcv49yhynyk.streamlit.app/](https://ai-study-agent-axvzmddspvrzcv49yhynyk.streamlit.app/)

## What it does

You ask a question. Instead of always doing the same thing, the agent first decides **which tool fits**, runs it, and only then writes an answer. It can also chain two tools together when a question needs both. The UI shows exactly which tools were used for every answer.

It also has **long-term memory**: tell it something about yourself (an exam date, a plan, a preference) and it can recall it later, even after a restart.

## Why this is different from a plain RAG chatbot

A [RAG chatbot](https://github.com/Iaimal/rag-notes-chatbot) always does the same fixed steps: retrieve from notes, then answer. This project's LLM instead **chooses an action** for each question, the same way a person would decide "let me check my notes" vs "let me do the math" vs "I already know this." This is the core idea behind AI agents: giving a model the ability to act, not just respond.

## Features

- Picks the right tool automatically: notes search, calculator, date lookup, or memory
- **Long-term memory:** saves personal facts and recalls them by meaning, across sessions
- Chains tools together for questions that need more than one (e.g. "search my notes for X, then calculate Y")
- Shows which tool(s) were used for every answer
- Falls back to answering directly when no tool is needed
- Handles malformed responses from the LLM without crashing
- A chat interface built with Streamlit

## How it works

1. **Decide:** the question is sent to the LLM (`openai/gpt-oss-20b` on Groq) along with a list of available tools. The model replies with a single JSON object naming which tool to use and what input to give it, or `"none"` if no tool is needed.
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
| `remember` | Saves a personal fact to long-term memory | "Remember that my exam is on December 5" |
| `recall` | Looks up saved facts by meaning | "When is my exam?" |

## Long-term memory

The agent keeps a persistent memory of facts the user tells it.

```
You:   Remember that my favorite subject is operating systems
Agent: Got it! Your favorite subject is Operating Systems.
       (restart the app or the script)
You:   What is my favorite subject?
Agent: Your favorite subject is operating systems.
```

**How it works**

- Each fact is turned into an embedding (`all-MiniLM-L6-v2`) and stored in a persistent ChromaDB collection (`memory_db/`).
- `recall` embeds the question and returns the closest memories by cosine similarity, ignoring weak matches below a minimum score (0.25). So "when is my exam?" finds "User's exam is on December 5" even though the words differ.
- Before saving, the agent checks for near-duplicates: a fact more than 90% similar to an existing one is not stored again.
- The agent loop stops after a single `remember` or `recall`, so a fact is never saved twice and no extra LLM call is wasted.

## Example questions

- "What is 23 times 47?" → uses `calculator`
- "What day is it today?" → uses `get_date`
- "How do I stop overfitting?" → uses `notes_search`
- "First search my notes for a way to reduce overfitting. Then calculate 3 times 4." → uses both `notes_search` and `calculator`
- "Remember that my exam is on December 5" → uses `remember`
- "When is my exam?" → uses `recall`
- "Tell me a joke" → uses no tool; the LLM answers directly

## Project structure

```
├── app.py            # Streamlit chat interface
├── agent.py          # Tool selection, tool execution, and the chaining loop
├── tools.py          # The calculator and get_date tool functions
├── search.py         # Notes retrieval (chunking, embedding, Chroma), reused from the RAG project
├── memory.py         # Long-term memory: save_memory() and recall() on a persistent Chroma store
├── notes/            # The text files notes_search can draw on
└── requirements.txt  # Python dependencies
```

`memory_db/` (the saved memories) is created at runtime and is not committed to the repo.

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
3. Set your Groq API key (free at console.groq.com/keys):
```
   set GROQ_API_KEY=your_groq_key
```
   On Mac or Linux, use `export GROQ_API_KEY=your_groq_key` instead.
4. Start the app:
```
   python -m streamlit run app.py
```
   It opens at `http://localhost:8501`.

To test the agent without the UI, run it from the project folder:
```
python -c "from agent import run_agent; print(run_agent('Remember that my exam is on December 5'))"
python -c "from agent import run_agent; print(run_agent('When is my exam?'))"
```

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repo, with `app.py` at the top level and the `notes/` folder included.
2. On share.streamlit.io, create a new app from that repo, with `app.py` as the main file.
3. Under **Advanced settings → Secrets**, add:
```
   GROQ_API_KEY = "your_groq_key"
```
4. Click **Deploy**.

Never commit your key to the repo. It belongs only in Streamlit's Secrets box or in an environment variable.

## A real bug I hit, and what it taught me

Early on, compound questions (e.g. "search my notes for X, then calculate Y") were failing silently — the agent would skip tools entirely and just answer from the LLM's own training knowledge, even though the prompt explicitly told it to always use a tool when relevant.

Debugging traced this to the model actually **planning both tool calls correctly**, but returning them as two separate JSON objects on two lines instead of one. My parsing code only expected a single JSON object, so it failed and silently fell back to "no tool needed" — making it look like the model had ignored the instructions, when it hadn't. The fix was to only read the model's first line of output per turn, and let the existing loop naturally ask again for the next step.

The lesson: when a model's behavior looks wrong, check the raw, unprocessed output before assuming the prompt or the model is at fault. The bug here was in how the response was being read, not in what the model produced.

## Another one: switching LLM providers

The hosted Hugging Face free tier ran out of credits and the agent started failing with `402 Payment Required`. I moved the LLM call to Groq, and because the provider call lives in a single `ask_llm()` function, the switch touched one function instead of the whole agent. The new model, `gpt-oss-20b`, is a reasoning model that spends part of its token budget on thinking, so I had to lower the reasoning effort and raise the token limits — otherwise it returned empty replies.

## Limitations

- **Memory resets on the free hosted app.** Streamlit Community Cloud has an ephemeral disk, so saved memories are lost when the app restarts. Memory persists fully when run locally, or in a container with a mounted volume.
- Memory is shared by everyone using one running instance — there is no per-user separation.
- Long-term memory stores facts, not the conversation: the agent doesn't remember earlier questions in the same chat unless they were saved with `remember`.
- Capped at 2 tool-use steps per question; a question genuinely needing three or more tool calls in sequence won't be fully resolved.
- Tool selection depends on the LLM correctly following the prompt's instructions, which isn't guaranteed for every phrasing of a question.
- Depends on a hosted model through an external API. Model and provider availability can change (Groq's model list has already changed once), which may require updating the model name in `agent.py`.
- The `calculator` tool uses `eval()` restricted to numbers and math symbols only, which is safe for this use case but not a general-purpose sandbox.
- The free hosted app goes to sleep after a period without visitors and takes a moment to wake up.

## Possible improvements

- Store memory in a hosted database so it survives restarts on the live app
- Per-user memory
- Let the user view and delete saved memories from the UI
- Raise `max_steps` and test with genuinely longer multi-tool chains
- Add more tools (e.g. web search, unit conversion)
- Log every tool decision for easier debugging and evaluation
- Add automated tests for the tool-selection logic

## Built with

Python · Streamlit · Groq (`gpt-oss-20b`) · sentence-transformers · ChromaDB

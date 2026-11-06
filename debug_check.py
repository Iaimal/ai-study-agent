from huggingface_hub import InferenceClient
import os
from agent import TOOLS_PROMPT

client = InferenceClient(model='Qwen/Qwen2.5-7B-Instruct', provider='featherless-ai', token=os.environ.get('HF_TOKEN'))
messages = [{'role': 'user', 'content': f'{TOOLS_PROMPT}\n\nQuestion: First search my notes for a way to reduce overfitting. Then calculate 3 times 4.'}]
result = client.chat_completion(messages=messages, max_tokens=60)
print(repr(result.choices[0].message.content))
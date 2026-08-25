import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model= 'openrouter:liquid/lfm-2.5-2.6b:free',
    temperature = 0.1
)


for chunk in model.stream('Hello, what is Python?'):
    print(chunk.text, end='', flush=True)


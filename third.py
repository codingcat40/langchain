import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage


load_dotenv()

model = init_chat_model(
    model= 'openrouter:liquid/lfm-2.5-2.6b:free',
    temperature = 0.1
)

conversation = [
    SystemMessage('You are a helpful assistant for questions regarding programming')
    ,
    HumanMessage('What is Python?'),
    AIMessage('Python is an Interpreted programming language'),
    HumanMessage('When was it released?')

]

response = model.invoke(conversation)

print(response)
print(response.content_blocks)
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

from langchain.chat_models import init_chat_model


load_dotenv()


@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str) -> str:
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()


@tool('locate_user', description="Look up a user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'Beijing'
        case 'XYZ123':
            return 'Chennai'
        case 'LLM123':
            return 'Kyoto'
        case _:
            return 'Unknown'


model = init_chat_model(
    model='openrouter:liquid/lfm-2.5-2.6b:free',
    temperature=0.3
)

checkpointer = InMemorySaver()


agent = create_agent(
    model="openrouter:liquid/lfm-2.5-2.6b:free",
    tools=[get_weather, locate_user],
    system_prompt="You are a helpful weather assistant, who always cracks jokes and is humorous while remaining helpful",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {'configurable': {'thread_id': 1}}

result = agent.invoke({
    'messages': [{"role": "user", "content": "What's the weather Like right now"}]},
    config=config,
    context=Context(user_id='XYZ123')
)

# print responses
print(result['structured_response'])
print(result['structured_response'].summary)
print(result['structured_response'].temperature_celsius)

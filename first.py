import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()


@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str) -> str:
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()


agent = create_agent(
    model="openrouter:liquid/lfm-2.5-2.6b:free",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke({

    "messages": [{"role": "user", "content": "What's the weather in Auroville right now"}]
})

print(result["messages"][-1].content_blocks)

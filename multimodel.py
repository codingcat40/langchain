from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

from langchain.messages import HumanMessage

load_dotenv()

model = init_chat_model('openrouter:nvidia/nemotron-nano-12b-v2-vl:free')

message = HumanMessage(content=[
        {'type': 'text', 'text': 'Describe the contents of this image'},
        {'type': 'image', 'url': 'https://ecfr.eu/wp-content/uploads/2026/05/595918088-scaled-1280x720-c-center.jpg'}
    ])

response = model.invoke([message])

print(response.content)
import os

from dotenv import load_dotenv

from langchain.agents import create_agent

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.tools import create_retriever_tool

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.chat_models import init_chat_model

load_dotenv()


raw_text = """
Auroville is an experimental township in Tamil Nadu, India, founded in 1968 by
Mirra Alfassa, known as "The Mother". It was designed by architect Roger Anger
and envisioned as a universal town where people from all countries could live
in peace and harmony, beyond all creeds, politics, and nationalities.

The town is organized around the Matrimandir, a large golden sphere used for
meditation, which sits at the symbolic center of the township. Auroville is
divided into four zones: the Residential Zone, the Industrial Zone, the
Cultural Zone, and the International Zone.

Auroville is also known for its reforestation and sustainable-living projects,
including renewable energy research, organic farming, and water table
restoration, turning what was once eroded wasteland into a thriving forest.
"""

# 2. Split into overlapping chunks so retrieval can find focused passages
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
documents = splitter.create_documents([raw_text])

# 3. Embeddings — free via the Hugging Face Inference API.

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGING_FACE_ACCESS_TOKEN"),
)

vector_store = FAISS.from_documents(documents, embedding=embeddings)

# similarity search test

# print(vector_store.similarity_search('Auroville is a place for sustainability'))
# print(vector_store.similarity_search('Who is the founder of Auroville', k=2))
# print(vector_store.similarity_search(
#     'what is matrimandir and what is its significance', k=2))


retriever = vector_store.as_retriever(search_kwargs={'k': 2})

retriever_tool = create_retriever_tool(
    retriever, name='kb_search', description='Search the natural places/ spiritual places/ retreat places in Indian subcontinent, database for the information')

agent = create_agent(model="openrouter:liquid/lfm-2.5-2.6b:free",
                     tools=[retriever_tool],
                     system_prompt=(
                         "You are a helpful assistant who answers questions on tourist places, spiritual destinations, retreats. "
                         "First call the kb_search retrieve tool to retrieve context, then answer succinctly. You may need to use it multiple times before answering."
                     )
                     )

result = agent.invoke({
    'messages': [{"role": "user", "content": "What is Auroville about and when and by whom it was founded"}]},
    
)
print(result)
print(result["messages"][-1].content)
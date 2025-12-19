from langchain_qdrant import QdrantVectorStore
from ollama import chat
from ollama import ChatResponse
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434",
)

vector_db=QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="learning_rag_qwen",  
)

user_query=input("Ask➡️➡️\t")

## make the similarity search
search_results=vector_db.similarity_search(
    query=user_query
)

context = "\n\n\n".join([
    f"Page Content: {result.page_content}\n"
    f"Page Number: {result.metadata['page_label']}\n"
    f"File Location: {result.metadata['source']}"
    for result in search_results
])


## system prompt
SYSTEM_PROMPT=f'''
                You are a helpfull AI Assistant who answeres user query based on the available context retrieved from a PDF file along with page_contents and page number.

                You should only ans the user based on the following context and navigate the user to open the right page number to know more.

                Context:
                    {context}
                
            '''

### chat model with ollama
# Initialize Ollama client
response: ChatResponse = chat(model='gemma3:1b', messages=[
  {
    'role': 'system',
    'content': SYSTEM_PROMPT,
  },
  {
    'role': 'user',
    'content': user_query,
  },
])

print(response.message.content)

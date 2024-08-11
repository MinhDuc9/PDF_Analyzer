import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Set the API key as an environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DOC_PATH = "notes.pdf"
CHROMA_PATH = "chroma"

loader = PyPDFLoader(DOC_PATH)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

# Define a prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

# Create the ChatOpenAI model instance
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o")

# Start interactive chat
print("Interactive Chat - Type 'exit' to quit")

while True:
    # Get user input
    query = input("You: ")
    if query.lower() == "exit":
        break

    # Retrieve context - top 5 most relevant (closest) chunks to the query vector
    docs_chroma = db_chroma.similarity_search_with_score(query, k=10)

    # Generate an answer based on given user query and retrieved context information
    context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

    # Load retrieved context and user query in the prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    # Call LLM model to generate the answer based on the given context and query
    response = model.invoke(prompt)

    # Print the response
    print(f"AI: {response.content}")
    print("\n" + "=" * 80 + "\n")

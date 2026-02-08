import chromadb
from langchain_google_genai import GoogleGenerativeAI
from prompts import SYSTEM_PROMPT, json_structure
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Initialize Chroma
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(
    name="inkind"
)

# User query
user_query = input("What do you want to know about growing vegetables?\n\n")

# Query Chroma
results = collection.query(
    query_texts=[user_query],
    n_results=1
)


print("Retrieved documents:", results["documents"])
print("Retrieved metadatas:", results["metadatas"])

# Build system-style prompt manually (Gemini doesn't have roles)
prompt = f"""
You are a helpful assistant. You answer questions about growing vegetables in Florida.
But you only answer based on knowledge I'm providing you.
You don't use your internal knowledge and you don't make things up.

If you don't know the answer, just say: I don't know

--------------------------

The data:
{results['documents']}

--------------------------

Question:
{user_query}
"""

# Initialize Gemini
llm = GoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
)

# Generate response
response = llm.invoke(prompt)

print("\n\n-----------------------------\n\n")
print(response)

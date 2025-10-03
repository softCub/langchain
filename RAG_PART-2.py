import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, "db","chroma_db")

embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")

db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)

question = "what is the name of the world "

retrive = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k":2, "score_threshold":0.5},
)

retrived_docs = retrive.invoke(question)
#retrived_docs = retrive.get_relevant_documents(question)
print("\n-----------retrived docs ------------------")

for i, doc in enumerate(retrived_docs,1):
    print(f"doc{i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"source:{doc.metadata.get('source')}\n")



import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings ,ChatGoogleGenerativeAI as ai

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir,"db","chroma_db")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(persist_directory=persistent_dir,embedding_function=embeddings)

question = "what was written in the diary"

retrive = db.as_retriever(
    search_type="similarity",search_kwargs={"k":3}
)
relavent_docs = retrive.invoke(question)

for doc in relavent_docs:
    print("\n-----------relavent docs ------------------")
    print(doc.page_content)

combined_input =(
    "here are some docs to help you answer the question\n"
    +question
    +"\n Relevant docs:\n"
    +"\n\n".join([doc.page_content for doc in relavent_docs])
    +"please provide a concise answer based on the above documents. If you don't know the answer, just say that you don't know, don't try to make up an answer."
)

model = ai(model="gemini-2.5-flash-lite")

Message = [
    SystemMessage("you are a helpful assistant that helps people find information."),
    HumanMessage(content=combined_input)
]
result = model.invoke(Message)
print("-----------result ------------------")
print(result.content)
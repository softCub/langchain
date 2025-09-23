from google.cloud import firestore
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI as ai
from langchain_google_firestore import FirestoreChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

projectID ="idkwha-968f3"
sessionId ="user_one_hmm"
collection_name="chat_history"

print("Initializing client")
client = firestore.Client(project=projectID)
print(client)

print("Initializing firestore history")
history =FirestoreChatMessageHistory(
    session_id=sessionId,
    collection=collection_name,
    client=client,)
print("done")
print("current chat history:",history.messages)

model = ai(model="gemini-2.5-flash-lite")

while(1):
    huminp = input("you:")
    if(huminp=="exit"):break
    history.add_user_message(huminp)

    aires= model.invoke(history.messages)
    history.add_ai_message(aires)

    print(history.messages)
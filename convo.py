from langchain_google_genai import ChatGoogleGenerativeAI as ai
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ai(model="gemini-2.5-flash-lite")

chatHistory=[]

SysMsg = SystemMessage(content="you are a pro gamer")

while(1):
    que=input("You: ")
    if(que=="exit"):break

    huMsg = HumanMessage(content=que)
    chatHistory.append(huMsg)

    resutl = model.invoke(chatHistory)
    print(resutl.content,"\n")
    chatHistory.append(AIMessage(content=resutl))
from langchain_google_genai import ChatGoogleGenerativeAI as ai
from langchain_core.messages import HumanMessage , SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm =ai(model="gemini-2.5-flash-lite")

message=[
    SystemMessage("you are a pro in gaming"),
    HumanMessage("what are the best games in the market "),
]
result =llm.invoke(message)
print(result.content)

from langchain_google_genai import ChatGoogleGenerativeAI as ai
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
llm=ai(model="gemma-3-27b-it")
result =llm.invoke("what is the root of 49")
print(result.content)
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI as ai
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model =ai(model="gemini-2.5-flash-lite")

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a facts expert how knows a lot about {animal}."),
    ("human","Tell me {count} facts."),
])

chain = prompt | model |StrOutputParser()
result =chain.invoke({"animal":"cat","count":2})
print(result)
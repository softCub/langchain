from langchain_google_genai import ChatGoogleGenerativeAI as ai
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ai(model="gemini-2.5-flash-lite")

template ="write a {tone} email to {company} expressing interest int the position, mentioning {skill} as key strength"

prompt = ChatPromptTemplate.from_template(template)
prompt=prompt.invoke({"tone":"soft","company":"samsung","position":"Ai engineer","skill":"ai"})
result = model.invoke(prompt)
print(result.content)

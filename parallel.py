from langchain_google_genai import ChatGoogleGenerativeAI as ai
from langchain.prompt import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ai(model="gemini-2.5-flash-lite")
summary = ChatPromptTemplate.from_messages(
    [
        ("system","You are a movie critic"),
        ("human","provide a brief summay on the topic {movid_name}")
    ]
)
def analyze_plot(plot):
    plot_temp = ChatPromptTemplate.from_messages(
        [
            ("system","you are a movie critic"),
            ("human","Analise the plot: {plot}. what are its strength and weaknesses"),
        ]
    )
def analyze_character(character):
    character_temp = ChatPromptTemplate.from_messages(
        [
            ("system","you are a movie critic"),
            ("human","Analise the characters {character}.what are their strengths and weaknesses")
        ]
    )


chain = (
    summary|model|StrOutputParser()|
    RunnableParallel()
)
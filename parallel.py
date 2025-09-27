from langchain_google_genai import ChatGoogleGenerativeAI as ai
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
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
    return plot_temp.format_prompt(plot=plot)
    
def analyze_character(character):
    character_temp = ChatPromptTemplate.from_messages(
        [
            ("system","you are a movie critic"),
            ("human","Analise the characters {character}.what are their strengths and weaknesses")
        ]
    )
    return character_temp.format_prompt(character=character)

def combine(plot_anal,character_anal):
    return f"Plot Analysis:\n{plot_anal}\n------------------------------------------\nCharacter Analysis:\n{character_anal}"
plot_branch =(
    RunnableLambda(lambda x:analyze_plot(x))|model|StrOutputParser()
)
character_branch=(
    RunnableLambda(lambda x:analyze_character(x))|model|StrOutputParser()
)
chain = (
    summary|model|StrOutputParser()|
    RunnableParallel(branches={"plot":plot_branch,"character":character_branch})|
    RunnableLambda(lambda x:combine(x['branches']['plot'],x['branches']['character']))
)
result =chain.invoke({"movid_name":"pasific rim"})
print(result)
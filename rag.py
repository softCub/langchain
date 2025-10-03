import os 
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from dotenv import load_dotenv

load_dotenv()

current_dir =os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir,"db","files","text.txt")
persistent_dir = os.path.join(current_dir,"db","chroma_db")

if not os.path.exists(persistent_dir):
    print("persistent dir missing!!")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"the file {file_path} not found")
    
loader = TextLoader(file_path)
documents = loader.load()
test_splitter = CharacterTextSplitter(chunk_size=10,chunk_overlap=0)
docs = test_splitter.split_documents(documents)

print("\n-----------documents ------------------")
print(f"no of document chunks:{len(docs)}")
print(f'sample chunk; {docs[0].page_content}')
# print(help(langchain_google_genai))

print("\n-----------embeddings ------------------")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
print("embeddings created")

print("\n-----------chroma vector store ------------------")
db = Chroma.from_documents(
    docs,embeddings,persist_directory=persistent_dir
)
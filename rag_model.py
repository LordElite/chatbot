# rag_model.py

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from openai import OpenAI
from langchain_community.tools import TavilySearchResults



# Load environment variables from the .env file
load_dotenv()

# Access your variables
langchain_key = os.getenv("LANGCHAIN_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# Initialize the OpenAI client
client = OpenAI(api_key=openai_key)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    
    
class RAGModel:
    def __init__(self):
        # Initialize the components
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = InMemoryVectorStore(self.embeddings)  #vector memory
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.tool = TavilySearchResults(   # Tavily search engine
            max_results=1,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
            # search_depth="advanced",
            # include_domains = [],
            # exclude_domains = []
            )

        # Load and chunk contents of the blog
        loader1 = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )

        loader = MergedDataLoader([loader1]) #line in case of loading several docuemnts firstly
        docs = loader.load()

        
        all_splits = self.text_splitter.split_documents(docs)  #splitting all the initial documents

        # Index chunks
        _ = self.vector_store.add_documents(documents=all_splits)

        # Define prompt for question-answering
        #self.prompt = hub.pull("rlm/rag-prompt")   #default prompt
        
        
        # Define state graph
        self.graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        self.graph_builder.add_edge(START, "retrieve")
        self.graph = self.graph_builder.compile()

    # Define retrieve method
    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    # Define generate method
    def generate(self, state: State):
        # Combine the retrieved document contents into a single context string
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        # Define the system message to establish the behavior of the LLM
        system_message = (
            "You are a gentle and polite research assistant. "
            "you don't need to answer huge results after user greeting expressions"
            "You provide helpful, professional, and accurate responses to user questions. "
            "Base your answers on the provided context and clearly state if something is not mentioned in the context."
            "If the context does not information to answer the question, you simply say I don't know"
        )

        # Prepare the messages in the format expected by the LLM
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Question: {state['question']}\n\nContext: {docs_content}"}
        ]

        # Generate a response from the LLM
        response = self.llm.invoke(messages)

        # Return the response as part of the state
        return {"answer": response.content}

    # Public method to get a response
    def get_response(self, question: str) -> str:
        initial_state = {"question": question, "context": [], "answer": ""}
        final_state = self.graph.invoke(initial_state)
        return final_state["answer"]
    
     #method to moderate statements
    def moderate_response(self, question: str) -> bool:
        response =client.moderations.create(
        model="omni-moderation-latest",
        input=question,
        )
        return response.results[0].flagged
    
    # Method to add new context and feed RAG
    def get_new_context(self, question: str) -> str:
        search = self.tool.invoke(question)
        url = search[0].get("url")
        new_loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer()
            ),
        )
        new_docs = new_loader.load()
        new_splits = self.text_splitter.split_documents(new_docs)  #splitting new documents
        # Index chunks
        _ = self.vector_store.add_documents(documents=new_splits)
        print(url)
        return question
    
    
 

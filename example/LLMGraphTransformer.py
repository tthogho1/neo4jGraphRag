from langchain_community.graphs import Neo4jGraph 
from langchain_openai import ChatOpenAI 
from langchain_experimental.graph_transformers import LLMGraphTransformer 
from PDFReader import load_pdf, split_text
 
llm = ChatOpenAI( 
    model= "gpt-3.5-turbo"
)

docs = load_pdf() 
tgt_chunks = split_text(docs) 
  
llm_transformer = LLMGraphTransformer(llm=llm) 
graph_documents = llm_transformer.convert_to_graph_documents(tgt_chunks) 

graph = Neo4jGraph() 
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

from langchain.chains import GraphCypherQAChain 
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI 
 
graph = Neo4jGraph() 

llm = ChatOpenAI(  
    model= "gpt-3.5-turbo"
) 

def add_limit_to_query(natural_language_query, limit=10):
    return f"{natural_language_query} (limit {limit})"

query_with_limit = add_limit_to_query("How many caches in infinispan?", limit=10)

client = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True) 
#result = client.invoke({'query': 'what is Named cache.?'}) 
# result = client.run('what is named cache.')
result = client.invoke({'query': query_with_limit})

print(result)
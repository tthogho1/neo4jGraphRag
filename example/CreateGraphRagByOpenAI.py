from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Neo4jVector
from dotenv import load_dotenv
import os

# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI

# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from neo4j import GraphDatabase

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI

load_dotenv()

text_path = "./text.txt"
loader = TextLoader(text_path)
raw_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)
documents = text_splitter.split_documents(raw_documents)


llm = OpenAI(model="gpt-3.5-turbo-instruct")  # Example LLM initialization
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# GraphDocument(nodes=[], relationships=[], source=Document(metadata={'source': './text.txt'}, page_content='Monkī Dī Rufi\nOfficial English Name:\nMonkey D. Luffy\nDebut:\nChapter 1;[1] Episode 1[2]\nAffiliations:\nStraw Hat Pirates;'))
# nodesが空のものを削除する
graph_documents = [doc for doc in graph_documents if doc.nodes]

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

# Initialize the graph instance
graph = Neo4jGraph(
    url=url,
    username=username,
    password=password,
)

for i, doc in enumerate(graph_documents):
    try:
        graph.add_graph_documents([doc], baseEntityLabel=True, include_source=True)
    except Exception as e:
        print(f"Error at index {i}: {e}")
        print(doc)

# Initialize embeddings
embeddings = OpenAIEmbeddings()

hybrid_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)
unstructured_retriever = hybrid_index.as_retriever()

driver = GraphDatabase.driver(uri=url, auth=(username, password))


def create_fulltext_index(tx):
    query = """
    CREATE FULLTEXT INDEX `fulltext_entity_id` 
    FOR (n:__Entity__) 
    ON EACH [n.id];
    """
    tx.run(query)


def create_index():
    with driver.session() as session:
        session.execute_write(create_fulltext_index)
        print("Fulltext index created successfully.")


try:
    create_index()
except:
    pass

driver.close()


class Entities(BaseModel):
    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

llmchat = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)

entity_chain = prompt | llmchat.with_structured_output(Entities)
entity_chain.invoke({"question": "Who are Luffy and Shanks?"})


def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke(question)
    for entity in entities.names:
        response = graph.query(
            """
        CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
        YIELD node, score
        WITH node
        CALL {
        WITH node
        MATCH (node)-[r]->(neighbor)
        WHERE type(r) <> 'MENTIONS'
        RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
        UNION ALL
        WITH node
        MATCH (node)<-[r]-(neighbor)
        WHERE type(r) <> 'MENTIONS'
        RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
        }
        RETURN output LIMIT 50
          """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join(el["output"] for el in response)
    return result


print(structured_retriever("Who is Luffy?"))


def full_retriever(question: str):
    structured_data = structured_retriever(question)
    unstructured_data = [
        el.page_content for el in unstructured_retriever.invoke(question)
    ]
    final_data = f"""Structured data:
    {structured_data}
    unstructured_data:
    {"#Document ". join(unstructured_data)}
    """
    return final_data


template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": full_retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke(input="Who is Luffy? Who are his companions?")
print(result)

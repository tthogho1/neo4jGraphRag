
import MeCab
from typing import List, Tuple
import traceback
import sys
import json
##from google.colab import userdata
from neo4j import GraphDatabase
#rom yfiles_jupyter_graphs import GraphWidget
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain.schema import Document
from retriever import utils.Retriever

llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125") 

graph = Neo4jGraph() #Neo4jGraphオブジェクトを作成

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),  # OpenAIの埋め込みモデルを使用してテキストをベクトル化
    search_type="hybrid",  # ベクトル検索とキーワード検索を組み合わせたハイブリッド検索を使用
    node_label="Document",  # 'Document'ラベルを持つノードを対象とする
    text_node_properties=["text"],  # 'text'プロパティの内容をベクトル化の対象とする
    embedding_node_property="embedding"  # ベクトル（埋め込み）を'embedding'プロパティに格納
)

class Entities(BaseModel):
    names: List[str] = Field(..., description="テキスト内に出現する全ての人物、組織エンティティ")

parser = PydanticOutputParser(pydantic_object=Entities)

prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are extracting organization and person entities from the text. Output should be in JSON format with a 'names' key containing a list of extracted entities."),
    ("human", "Extract entities from the following input: {question}")
])

prompt1 = prompt1.partial(format_instructions=parser.get_format_instructions())

def extract_entities(text: str) -> Entities:
    try:
        messages = prompt1.format_messages(question=text)
        response = llm(messages)
        entities = parser.parse(response.content)
        
        return entities

    except Exception as e:
        print(f"エンティティ抽出中にエラーが発生しました: {e}")
        return Entities(names=[])


def generate_full_text_query(input: str) -> str:
    tagger = MeCab.Tagger()
    nodes = tagger.parseToNode(input)
    
    important_words = []
    while nodes:
        if nodes.feature.split(',')[0] in ['名詞', '動詞', '形容詞']:
            important_words.append(nodes.surface)
        nodes = nodes.next
    
    if not important_words:
        return input
    
    return ' OR '.join(f'"{word}"' for word in important_words)



def structured_retriever(question: str) -> str:
    try:
        entities = extract_entities(question)
        if not entities.names:
            return "質問に関連するエンティティが見つかりませんでした。"
        
        result = ""
        for entity in entities.names:
            query = generate_full_text_query(entity)
            print(query)
            if query:
                try:
                    response = graph.query(
                        """CALL db.index.fulltext.queryNodes('entity', $query, {limit:20})
                        YIELD node,score
                        CALL {
                          WITH node
                          MATCH (node)-[r:!MENTIONS]->(neighbor)
                          RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                          UNION ALL
                          WITH node
                          MATCH (node)<-[r:!MENTIONS]-(neighbor)
                          RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                        }
                        RETURN output LIMIT 1000
                        """,
                        {"query": query},
                    )
                    result += "\n".join([el['output'] for el in response])
                except Exception as e:
                    print(f"クエリ実行中にエラーが発生しました: {e}")
        
        return result if result else "関連情報が見つかりませんでした。"
    except Exception as e:
        print(f"エンティティ抽出中にエラーが発生しました: {e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_last = traceback.extract_tb(exc_traceback)[-1]
        traceback.print_exc()
        return "エンティティの抽出中にエラーが発生しました。"


def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
    {structured_data}
    Unstructured data:
    {"#Document ". join(unstructured_data)}
    """
    print(final_data)
    return final_data

#_search_query = RunnableLambda(lambda x: x["question"])

template = """あなたは優秀なAIです。下記のコンテキストを利用してユーザーの質問に丁寧に答えてください。
必ず文脈からわかる情報のみを使用して回答を生成してください。
コンテキストに関連情報がない場合は、その旨を述べた上で一般的な回答を提供してください。

コンテキスト:
{context}

ユーザーの質問: {question}"""

prompt = ChatPromptTemplate.from_template(template)
def process_query(user_query):
    # 1. コンテキストの取得
    #search_results = _search_query(user_query)
    context = retriever(user_query)
    
    # 2. 入力データの準備
    inputs = {
        "context": context,
        "question": user_query
    }
    
    # 3. プロンプトの適用
    formatted_prompt = prompt.format(**inputs)
    
    # 4. LLMへの問い合わせ
    response = llm.predict(formatted_prompt)
    
    # 5. 結果の出力（StrOutputParserの代わり）
    return response

# 使用例
user_question = "めぐみんを好きな人がいますか？"
result = process_query(user_question)
print(result)

#graph._driver.close()

# めぐみんの好きなことは爆裂魔法です。

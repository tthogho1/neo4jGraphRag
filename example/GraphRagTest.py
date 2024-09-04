import os
import re
import wikipedia
import MeCab
from typing import List, Tuple
#from google.colab import userdata
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

#MeCabを使用してテキストを形態素解析し、固有表現を抽出する関数を定義
def tokenize_and_extract_entities(text: str) -> Tuple[List[str], List[str]]:
    """
    テキストを形態素解析し、トークンと固有表現を抽出する
    
    :param text: 解析対象のテキスト
    :return: トークンのリストと固有表現のリスト
    """
    tagger = MeCab.Tagger()
    parsed = tagger.parse(text)
    
    tokens = []
    entities = []
    
    for line in parsed.split('\n'):
        if line == 'EOS':
            break
        
        parts = line.split('\t')
        if len(parts) < 2:
            continue  # 無効な行はスキップ
        
        surface = parts[0]
        feature = parts[1]
        
        features = feature.split(',')
        
        tokens.append(surface)
        
        # 固有名詞の抽出
        if len(features) > 1 and features[0] == '名詞' and features[1] in ['固有名詞', '人名', '組織', '地名']:
            entities.append(surface)
    
    return tokens, entities

# テキスト分割関数の定義
def split_text(text: str, max_length: int = 1000) -> List[str]:
    """
    テキストを指定された最大長で分割する
    
    :param text: 分割対象のテキスト
    :param max_length: 分割後の最大文字数
    :return: 分割されたテキストのリスト
    """
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]


article = "この素晴らしい世界に祝福を"

# Wikipediaの日本語記事を用いる
wikipedia.set_lang("ja")
page = wikipedia.page(article, auto_suggest=False)
content = re.sub('（.+?）', '', page.content)  # ふりがなを除去

# テキストを分割
chunks = split_text(content)

all_tokens = []
all_entities = []

for chunk in chunks:
    try:
        tokens, entities = tokenize_and_extract_entities(chunk)
        all_tokens.extend(tokens)
        all_entities.extend(entities)
    except Exception as e:
        print(f"チャンク処理中にエラーが発生しました: {e}")
        continue

# MeCabで処理したトークンとエンティティを使用して、より適切なDocumentオブジェクトを作成
processed_content = " ".join(all_tokens)  # トークンを空白で結合

# エンティティ情報をメタデータとして追加
metadata = {"entities": list(set(all_entities))}  # 重複を除去

# Documentオブジェクトの作成
raw_documents = [Document(page_content=processed_content, metadata=metadata)]

# チャンクサイズの調整（必要に応じて）
text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=64)

# ドキュメントを分割
documents = text_splitter.split_documents(raw_documents)

llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125") #モデルはお好みで指定
llm_transformer = LLMGraphTransformer(llm=llm) #LLMTransformerオブジェクトを作成

#ドキュメントをグラフに変換。documents全部入れると時間もコストもかかるので、documents[0:10]など小規模で試してみると良い
graph_documents = llm_transformer.convert_to_graph_documents(documents[0:10])

graph = Neo4jGraph() #Neo4jGraphオブジェクトを作成

#Neo4jGraphオブジェクトにドキュメントを追加
graph.add_graph_documents(
    graph_documents, 
    baseEntityLabel=True,
    include_source=True
)

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),  # OpenAIの埋め込みモデルを使用してテキストをベクトル化
    search_type="hybrid",  # ベクトル検索とキーワード検索を組み合わせたハイブリッド検索を使用
    node_label="Document",  # 'Document'ラベルを持つノードを対象とする
    text_node_properties=["text"],  # 'text'プロパティの内容をベクトル化の対象とする
    embedding_node_property="embedding"  # ベクトル（埋め込み）を'embedding'プロパティに格納
)
graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

class Entities(BaseModel):
    names: List[str] = Field(..., description="テキスト内に出現する全ての人物、組織エンティティ")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting organization and person entities from the text. Output should be in JSON format with a 'names' key containing a list of extracted entities."),
    ("human", "Extract entities from the following input: {question}")
])

parser = PydanticOutputParser(pydantic_object=Entities)
entity_chain = prompt | llm | parser

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
        entities = entity_chain.invoke({"question": question})
        if not entities.names:
            return "質問に関連するエンティティが見つかりませんでした。"
        
        result = ""
        for entity in entities.names:
            query = generate_full_text_query(entity)
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
    return final_data

_search_query = RunnableLambda(lambda x: x["question"])

template = """あなたは優秀なAIです。下記のコンテキストを利用してユーザーの質問に丁寧に答えてください。
必ず文脈からわかる情報のみを使用して回答を生成してください。
コンテキストに関連情報がない場合は、その旨を述べた上で一般的な回答を提供してください。

コンテキスト:
{context}

ユーザーの質問: {question}"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke({"question": "めぐみんの好きなことは？"})
# めぐみんの好きなことは爆裂魔法です。

import os
import re
import wikipedia
import MeCab
from typing import List, Tuple
from neo4j import GraphDatabase
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






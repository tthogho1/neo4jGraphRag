from flask import Blueprint, render_template,url_for, request

import re
import wikipedia
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain.schema import Document

from utils.TextOperation import tokenize_and_extract_entities, split_text

# Blueprintを作成
create_bp = Blueprint('create', __name__)

@create_bp.route('/')
def hello():
    return render_template('index.html')

@create_bp.route('/create_page', methods=['POST'])
def create_page():
    article = request.form.get('wiki_title', '')  # フォームからデータを取得
    
    if not article:
        return "タイトルが入力されていません", 400  # エラーレスポンス

    # Wikipediaの日本語記事を用いる
    wikipedia.set_lang("ja")
    page = wikipedia.page(article, auto_suggest=True)
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
        
    
    return render_template('index.html')

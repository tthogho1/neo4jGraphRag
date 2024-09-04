from flask import Blueprint, render_template, jsonify, request
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
from utils.Retriever import retriever


query_bp = Blueprint('query', __name__)

llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125") 

@query_bp.route('/query')
def query_view():
    return render_template('query.html')

@query_bp.route('/execute', methods=['POST'])
def query_execute():
    user_question  = request.form.get('question', '')
    
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

    result = process_query(user_question)
    
    # JSON形式でレスポンスを返す
    return jsonify({"answer": result})

import MeCab
import re

from typing import List, Tuple

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

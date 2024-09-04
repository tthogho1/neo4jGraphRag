import MeCab

mecab = MeCab.Tagger() 

result = mecab.parse("すもももももももものうち")

print(result)
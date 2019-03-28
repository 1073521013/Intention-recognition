import os
import jieba
from starspace import EmbeddingIntentClassifier
from countvec import CountVectorsFeaturizer
# from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
# jieba.load_userdict('my_dir/dir.txt')
path = 'file_txt'
files = os.listdir(path)
x=[]
y=[]
for file in files:
    if not os.path.isdir(file):
        f = open(path+'/'+file,encoding='utf8')
        iter_f = iter(f)
        for line in iter_f:
            if 'text' in line:
                pass
            elif line.strip() == "":
                continue
            else:
                str_ = line.strip()
                li = str_.split(' ')
                # t=' '.join(jieba.cut(li[0]))
                # x.append(t)
                x.append(li[0])
                y.append(li[1])

# print(x)
# vect = CountVectorizer(token_pattern=r'(?u)\b\w\w+\b', strip_accents=None,stop_words=None, ngram_range=(1,1),max_df=1, min_df=1, max_features=None)
# x = vect.fit_transform(x).toarray()
vec=CountVectorsFeaturizer()
x=vec.train(x)
vec.persist('vec')

out = EmbeddingIntentClassifier()
out.train(x,y)
out.persist('runs')
from training_data import Message
import numpy as np
from starspace import EmbeddingIntentClassifier
from countvec import CountVectorsFeaturizer
out = EmbeddingIntentClassifier()
vec=CountVectorsFeaturizer()
component=out.load('runs')
vect = vec.load('vec')
import codecs
def default_output_attributes():
    return {"intent": {"name": "", "confidence": 0.0}}

def accuracy(pred, actual):
    pred = np.array(pred)
    actual = np.array(actual)
    """Returns percentage of correctly classified labels"""
    return sum(pred==actual) / len(actual)

y=[]
y_=[]
for line in  codecs.open('test.txt','r',encoding='utf8'):
    x = line.strip()
    x = x.split('|')
    x_ = x[1]
    y.append(int(x_[6:]))
    message = Message(str(x[0]))
    x = vect.process(message)
    result = component.process(x)
    y0=result[0]
    print(y0)
    y_.append(int(y0['name']))
print('pred: ',y_)
print('ture: ',y)
print(accuracy(y,y_))

# text='我想在香草园买点吃的，请问有什么地方可以推荐吗？'
# message = Message(text)
# x= vect.process(message)
# result=component.process(x)
# print(result)

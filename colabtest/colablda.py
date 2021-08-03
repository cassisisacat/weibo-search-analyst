#!pip install gensim
#!pip install numpy
#!pip install scipy
#!pip install jieba

from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import codecs
import os
train=[]

fp=codecs.open('/content/drive/MyDrive/python-weibo-lda/weibo-search/weibo/data_keywords.dat','r',encoding='utf-8')
for line in fp:
  if line !='':
    line=line.split()
    train.append([w for w in line])

dictionary = corpora.Dictionary(train)

corpus =[dictionary.doc2bow(text) for text in train]

lda=LdaModel(corpus=corpus, id2word=dictionary,num_topics=20, passes=60)

for topic in lda.print_topics(num_words=20):
  termNumber=topic[0]
  print(topic[0],':',sep='')
  listOfTerms=topic[1].split('+')
  for term in listOfTerms:
    listItems=term.split('*')
    print(' ',listItems[1],'(',listItems[0],')',sep='')

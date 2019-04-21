#coding=utf-8

import numpy as np
import codecs
import json
def loadData(filePath):
        f = codecs.open(filePath,'r', 'utf-8')
        data = {'newsId':[],'title':[],'coreEntityEmotions':[],'content':[],'keywords':[]}
        #data = []
        for line in f.readlines():
            news = json.loads(line.strip())
            #data.append(news)
            data['title'].append(news['title'])
            data['content'].append(news['content'])           
        return data
        
data = loadData('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/data/coreEntityEmotion_example.txt')

import jieba

### 直接提取关键词
import jieba.analyse
# 关键词提取
# 导入停用词表
stopwords = []
with open('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/中文停用词表.txt', "r", encoding='UTF-8') as text:
    stopwords.extend(text.read().split())

with open('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/哈工大停用词表.txt', "r", encoding='UTF-8') as text:
    stopwords.extend(text.read().split())

with open('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/StopWords.txt', 'w', encoding='utf-8') as f:
    for s in stopwords:
        f.write(s+'\n')
### title entity
#jieba.set_dictionary('data/dict.txt.big')
jieba.analyse.set_stop_words('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/StopWords.txt')
data['titleentity'] = []
tag = []
for content in data['title']: 
    tags = jieba.analyse.extract_tags(content, topK=5, withWeight=True, allowPOS=('n', 'nr', 'ns', 'nt', 'nz'))
    tag.append(tags)
    entity = []
    for t, w in tags:
        if w >= 1.0:
            entity.append(t)
    data['titleentity'].append(entity)
#print(",".join(tags))
with open('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/titleentity.txt','w') as f:
    for s in range(len(data['title'])):
        f.write(" ".join(data['titleentity'][s])+"\n")

### content entity
jieba.analyse.set_stop_words('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/StopWords.txt')
data['keywords'] = []
tag = []
for content in data['content']:
    #sentense = " ".join(content.split()) 
    tags = jieba.analyse.extract_tags(content, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns', 'nt', 'nz'))
    tag.append(tags)
    entity = []
    for t, w in tags:
        if w >= 0.5:
            entity.append(t)
    data['keywords'].append(entity)
#print(",".join(tags))
with open('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/keywords.txt','w') as f:
    for s in range(len(data['title'])):
        f.write(" ".join(data['keywords'][s])+"\n")
with open('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/data/keywords.txt','r') as f:
    for i in f:
        data['keywords'].append(i.split(" "))
## 直接计算 tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_Vectorizer = TfidfVectorizer()

document = []
for i in data['keywords']:
   i = " ".join(i)
   document.append(i)
tfidf = tfidf_Vectorizer.fit_transform(document)
del document
s = tfidf.toarray()
"""
### 提取词频，去停用词，计算 tf-idf

# jieba 分词
for content in data['searchwords']:
    sentense = " ".join(content)
    tags = jieba.lcut(sentense)
    data['keywords'].append(tags)
#print(",".join(tags))
    
# 出现次数最多的作为停用词
word_freq = {}
for word in data['keywords']:    
    for i in word:
        if i in word_freq:
            word_freq[i] += 1
        else:
            word_freq[i] = 1
freq_word = []
for word, freq in word_freq.items():
    freq_word.append((word, freq))
freq_word.sort(key = lambda x: x[1], reverse = True)
max_number = 50
freq_word[: max_number]
# 导入停用词表
stopwords = []
with open('C:/Users/zhaohao/Desktop/python/data_analysis/sougou_analysis/中文停用词表.txt', "r", encoding='UTF-8') as text:
    stopwords.extend(text.read().split())

with open('C:/Users/zhaohao/Desktop/python/data_analysis/sougou_analysis/哈工大停用词表.txt', "r", encoding='UTF-8') as text:
    stopwords.extend(text.read().split())

for s in freq_word[: max_number]:
    stopwords.append(s[0])

with open('C:/Users/zhaohao/Desktop/python/data_analysis/sougou_analysis/StopWords.txt', 'w', encoding='utf-8') as f:
    for s in stopwords:
        f.write(s+'\n')

def filterStopWords(line, stopwords):
    for i in line:
        if i in stopwords:
            line.remove(i)
    return line


# 计算 TF_IDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# 01、构建词频矩阵，将文本中的词语转换成词频矩阵
vectorizer = CountVectorizer()
# a[i][j]:表示j词在第i个文本中的词频
document = []
for i in data['keywords']:
    i = filterStopWords(i, stopwords)
    i = " ".join(i)
    document.append(i)
X = vectorizer.fit_transform(document)
# 02、构建TFIDF权值
transformer = TfidfTransformer()
# 计算tfidf值
tfidf = transformer.fit_transform(X)
# 03、获取词袋模型中的关键词
word = vectorizer.get_feature_names()
# tfidf矩阵
s = tfidf.toarray()
"""


### 训练测试

from sklearn.cross_validation import train_test_split
train_X,test_X, train_y, test_y = train_test_split(s,data['gender'],test_size = 0.3,random_state = 0)

from sklearn import svm
clf = svm.SVC(gamma='auto', decision_function_shape='ovr')
clf.fit(train_X, train_y) 
# # 查看分类器数目 
# dec = clf.decision_function([[1]])
# dec.shape[1]

# from xgboost import XGBClassifier
# clf = XGBClassifier(objective='multi:softmax')
# clf.fit(train_X, train_y)

# test accuracy
pre_y = clf.predict(test_X)
testy = np.array(test_y)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(testy, pre_y)
print(accuracy)
# count = 0
# yy = np.hstack((pre_y.reshape(-1,1), testy.reshape(-1,1)))# 按行拼接
# for y in yy:
    # if y[0] == y[1]:
        # count+=1
    # else:
        # pass
# accuary = count*1.0/len(pre_y)


#coding=utf-8

import numpy as np
import codecs
import json


"""
### 合并多个停用词表
# 导入停用词表
stopwords = []
with open('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/中文停用词表.txt', "r", encoding='UTF-8') as text:
    stopwords.extend(text.read().split())

with open('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/哈工大停用词表.txt', "r", encoding='UTF-8') as text:
    stopwords.extend(text.read().split())

with open('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/StopWords.txt', 'w', encoding='utf-8') as f:
    for s in stopwords:
        f.write(s+'\n')
"""
        
### title + contnet entity
import jieba
### 直接提取关键词
import jieba.analyse
# 正则表达式，用于切分文本
import re
import nltk
f = codecs.open('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/data/coreEntityEmotion_example.txt','r', 'utf-8')
fi = f.readlines()
f.close()
jieba.analyse.set_stop_words('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/StopWords.txt')
tag2 = []
"""
import nltk ### 只对英文有效
nltk.download()### 缺少包的时候使用
s = 'I love natural language processing technology!'
s_token = nltk.word_tokenize(s)
s_tagged = nltk.pos_tag(s_token)
#s_ner = nltk.chunk.ne_chunk(s_tagged)
#print(s_ner)

### 对词性筛选
for tup in pos_result:
    word = tup[0]
    pos_word = tup[1]
    if pos_word in pos or word in raw_nn:
        key_word_list.append(word)
"""
count = 0
pos= ['NN','NNS','NNP','NNPS']
### 计算 nltk 的词频  
def nltk_cipin(tags):
    word_freq = {}
    for word in tags:    
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    freq_word = []
    for word, freq in word_freq.items():
        freq_word.append((word, freq))
    freq_word.sort(key = lambda x: x[1], reverse = True)
    max_number = 3
    words=[]
    for word, freq in freq_word[: max_number]:
        words.append(word)
    return words

    for line in fi:
#                count += 1
#                if count == 600:
#                    assert False
        news = json.loads(line.strip()) 
        i= news['title']+"。"+news['content']
        sentences = re.split('[,，。！？?、‘’“”:（）(){}‘;:<>《》/ ]'.encode('utf-8').decode('utf-8'), i.strip())
        for sentence in sentences:
            if sentence == "":
                sentences.remove(sentence)
        tags = jieba.analyse.extract_tags(" ".join(sentences), topK=3, withWeight=True, allowPOS=('n', 'nr', 'ns', 'nt', 'nz'))
        """
        目前解决不了中英混合文本单独提取
        """
        if len(tags) < 1:### jieba 提取不了说明是全英文文本，用 nltk
            s_token = nltk.word_tokenize(" ".join(sentences))
            s_tagged = nltk.pos_tag(s_token)
            tags = []
            for tup in s_tagged:
                word = tup[0]
                pos_word = tup[1]
                if pos_word in pos:
                    tags.append(word.lower())
                tags = nltk_cipin(tags) ### 提取出出现次数前三的作ner
        else: ### jieba 提取出来说明是含大部分中文的文本  
            entity = []
            for t, w in tags:
                if w >= 0.5:
                    entity.append(t)
            if len(entity) == 0:
                entity = [tags[0][0]]
            tags = entity
        tag2.append(tags)
    with codecs.open('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/contents.txt','w','utf-8') as f2:    
        for tags in tag2:
            f2.write(" ".join(tags))
            f2.write('\n')         
        
"""
试验的其他方法，效果不理想
"""
"""
### title entity
#jieba.set_dictionary('data/dict.txt.big')
jieba.analyse.set_stop_words('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/StopWords.txt')
data['titleentity'] = []
tag1 = []
for content in data['title']: 
    tags = jieba.analyse.extract_tags(content, topK=3, withWeight=True, allowPOS=('n', 'nr', 'ns', 'nt', 'nz'))
    #tag1.append(tags)
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
data['keywords'] = []
tag2 = []
for content in data['content']:
    #sentense = " ".join(content.split()) 
    tags = jieba.analyse.extract_tags(content, topK=3, withWeight=True, allowPOS=('n', 'nr', 'ns', 'nt', 'nz'))
    #tag2.append(tags)
    entity = []
    for t, w in tags:
        if w >= 0.5:
            entity.append(t)
	
    data['keywords'].append(entity)
#print(",".join(tags))
with open('C:/Users/zhaohao/Desktop/python/data_analysis/souhuanalysis/keywords.txt','w') as f:
    for s in range(len(data['title'])):
        f.write(" ".join(data['keywords'][s])+"\n")
""" 
    
    
###    
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
import jieba.posseg as pseg
# jieba 分词
for content in data['searchwords']:
    sentense = " ".join(content)
    tags = pseg.cut(sentense)
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


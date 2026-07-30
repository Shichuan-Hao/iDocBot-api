# 准备文本数据

import jieba
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

sentences = ["我喜欢吃苹果", "苹果是我最喜欢吃的水果", "我喜欢用苹果手机"]

# 使用 jieba 进行中文分词
sentences_splits = []  # 创建一个空的大列表，是一个二维列表（列表的列表），例如：[ ['我', '喜欢', '编程'], ['自然语言', '处理', '很', '有趣'] ]。这种结构非常适合后续喂给 Word2Vec 模型。
for sentence in sentences:  # 遍历列表中的每一句话（每一次拿一个字符串）
    words = list(jieba.cut(sentence)) # 对这句话进行分词，得到这个词列表
    sentences_splits.append(words)  # 把这个词的列表放进大列表里

# 上面的循环等价于，Python 的列表推导式
# tokenized_sentences = [list(jieba.cut(sentence)) for sentence in sentences]
# 打印查看
print(sentences_splits)
# 输出：[['我', '来到', '北京', '清华大学'], ['他', '喜欢', '自然语言', '处理']]

# 训练Word2Vec模型
model = Word2Vec(sentences_splits, vector_size=100, window=5, min_count=1, workers=2)

# 获取每个句子的平均向量表示
def get_sentence_vector(sentence, model):
    words = list(jieba.cut(sentence))
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return sum(word_vectors) / len(word_vectors)

sentence_vectors = [get_sentence_vector(sentence, model) for sentence in sentences]

# 计算句子间的余弦相似度
cosine_similarities = cosine_similarity(sentence_vectors)

print(cosine_similarities)

print("="*60)

print(sentence_vectors)

print("="*60)

print(sentence_vectors[0])

print("="*60)

print(len(sentence_vectors[0]))

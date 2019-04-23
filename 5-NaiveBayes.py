import numpy as np

# 加载features labels
def loadDataset():
    # 文档样本
    features = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 对应的标签 0代表正常文本  1代表侮辱性文本
    labels = [0,1,0,1,0,1]
    return features, labels


# 统计所有文档中的出现过的所有的词条列表
def collectWord(dataset):
    wordSet = set([])
    # 遍历文档中每一段话
    for document in dataset:
        # 先将document转化为set的形式，确保其中元素的唯一性
        # 将预先准备的wordSet和set取并集！
        wordSet = wordSet | set(document)
    # 将并集还原为list形式，方便操作！
    return list(wordSet)

# 根据词条列表中的词是否出现在文档中样本中，将由单词构成的词条样本转化为01构成的数字样本
def setWord2Vec(wordSet,features):
    # 新建一个长度为和wordSet一样长，各维度的元素值为0的列表
    returnVector = [0]*len(wordSet)
    # 遍历文档中的每一个词条
    for word in features:
        if word in wordSet:
            returnVector[wordSet.index(word)] = 1
        else:
            print('the word [{}] is not in my vocabulary!'.format({word}))
    return returnVector


def trainNB0(features, labels):
    # 获取数据-特征集中的样本的数目
    num_samples = len(features)
    # 获取构成样本的特征的数目
    num_features = len(features[0])
    # 计算出所有的文本中  1 所占的比例
    proportion = sum(labels)/float(num_samples)
    # 创建一个长度为词条向量等长的列表
    p0Num = np.zeros(num_features)
    p1NUm = np.zeros(num_features)
    p0Denom = 0.0
    p1Denom = 0.0

    for i in range(num_samples):
        # 若该词条对应的标签为1
        if labels[i] == 1:
            p1NUm += features[i]
            p1Denom +=sum(features[i])
        else:
            p0Num += features[i]
            p0Denom += sum(features[i])


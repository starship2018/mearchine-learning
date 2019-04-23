from math import log

# 计算熵
# 熵相对于信息的价值，一个数据集在划分前和划分后的信息熵是不一样的，牛逼的划分可能会使得信息熵大增
# 决策树就是这样的一种寻找最合理的，使得信息熵增益极大的特征来划分数据集的算法！
def calculateEnt(dataset):
    # 计算数据集中的样本的个数
    count_samples = len(dataset)
    # print(count_samples)
    # 创建出字典来统计标签种类及其数目
    labelCount = {}
    # 针对每一行数据，有：
    for line in dataset:
        label = line[-1]
        # 若暂未存入当前标签
        if label not in labelCount.keys():
            # 新建键值对以存储标签及其数目
            labelCount[label] = 0
        # 注意！这里不要使用else进行判断，因为若发现标签未被存入时，先设定其计数为0，然后给其加上1.若使用了else就只是给其计数为0，没有加上1，那么结果会导致所有的
        # 标签统计数会比真实值少1！
        labelCount[label] += 1
    # 初始化熵为0
    Ent = 0.0
    # 计算各个标签的熵
    for key in labelCount:
        # 计算各个标签出现的频率
        prob = float(labelCount[key]/count_samples)
        # print(prob)
        # 计算熵
        Ent -= prob * log(prob, 2)
    return Ent


# 创建数据集
def createDataset():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 划分数据集【根据第no_feature号特征的不同取值，将数据集划分为不同的分支！】
# 找出特征序号为no_feature，值为value的样本，把这个特征从数据中剔除，重新组成新的数据集，这个数据集中的no_feature特征的是都为value，虽然已经被剔除了！
def splitDataset(dataset, no_feature, value):
    # python 在传递列表的时候，直接传递的是列表的引用。也就是说在函数内对传入的列表的修改会影响到原始的列表本身！
    # 这里我们创建出一个新的数据集，为的是不对原始数据集产生实际上的修改！
    resDataset = []

    for line in dataset:
        if line[no_feature] == value:
            # 这里解释一下append和extend的区别：
            # append是在末尾将参数作为一个元素进行添加，而extend是将参数中的每一个元素都逐个进行添加
            reducedFeatureVec = line[:no_feature]
            reducedFeatureVec.extend(line[no_feature+1:])
            resDataset.append(reducedFeatureVec)

            # forwardDataset = line[:no_feature].extend(line[no_feature+1:])
            # resDataset.append(forwardDataset)
    return resDataset


# 如何选出最好的数据集的划分情况呢？
def choosefeaturesplit(dataSet):
    # 获取特征数
    numFeatures = len(dataSet[0]) - 1
    # 原始数据集的信息熵
    baseEntropy = calculateEnt(dataSet)
    # 最大的信息熵
    bestEnt = 0.0
    # 最优的特征
    best_no_Feature = -1
    # 对数据集中的各个特征进行遍历，虚拟若以这个特征划分数据集所得到的信息熵会是多少？
    for i in range(numFeatures):
        # i号特征的取值集合【都是不同的】
        # 利用的set集合的元素都是不相同的！
        featList = set([line[i] for line in dataSet])
        # 信息增益
        newEntropy = 0.0
        # 现在，对第i号特征的所有取值，分别进行划分数据集
        for value in featList:
            subDataset = splitDataset(dataSet, i, value)
            # 计算对value划分出的数据占总数据集的比例
            prop = len(subDataset)/float(len(dataSet))
            newEntropy += prop * calculateEnt(subDataset)
        # 计算出按此特征划分数据集的信息增益
        infoGain = baseEntropy - newEntropy
        # 若是以i号特征划分数据集的信息增益比原来的信息增益还要大，则将最佳增益划分特征号给当前的i，同时最佳信息增益的值为当前值！
        if infoGain > baseEntropy:
            bestEnt = infoGain
            best_no_Feature = i
    return best_no_Feature

# 当遍历完所有的特征之后，类标签任然不唯一，也就是说分支下面还存在着不同分类的实例
# 我们采用的是多数表决的方法来完成最后的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 字典：classCount和classCount.items()的区别！classCounta默认是classCount.keys()!
    # sorted函数：对迭代器进行排序！默认是升序排列！
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    # 返回投票数最多的票名！即类名
    return sortedClassCount[0][0]


def createDecisionTree(dataset, features):   # 参数是数据集和数据特征名列表
    # 获取数据集中的最后一列的类标签，并存入classList列表
    classList = [sample[-1] for sample in dataset]
    # 正常情况下，这个类标签列表中的类应该是同一个类，但是可能也会出现异常情况
    # 那么我们来判断一下

    # 若数据集都属于同一个类，那么返回这个类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    if len(dataset[0]) == 1:
        return majorityCnt(classList)

    # 确定最优的分类特征
    bestFeatureNo = choosefeaturesplit(dataset)
    # 得到最优特征的名字
    bestFeature = features[bestFeatureNo]

    # 使用字典潜逃字典的方式存储分类树信息
    myTree = {bestFeature: {}}

    # 新建特征标签列表
    newLabels = features[:]
    # 删除特征列表中当前分类的数据特征
    del newLabels[bestFeatureNo]
    # 获取数据集中最优特征所在的列
    bestFeatureUniqueValue = set([sample[bestFeatureNo] for sample in dataset])
    # 遍历每一个特征取值
    for value in bestFeatureUniqueValue:
        # 使用递归的方法借助该特征值对数据集进行分类
        myTree[bestFeature][value] = createDecisionTree(splitDataset(dataset, bestFeatureNo, value), newLabels)
    return myTree


myDat, labels = createDataset()
myTree = createDecisionTree(myDat, labels)
print(myTree)

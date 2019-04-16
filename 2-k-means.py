import numpy as np
import random
import math
from matplotlib import pyplot as plt

# 加载数据集
def loadData(fileName):
    data = []
    fr = open(fileName, 'r')
    for line in fr.readlines():
        curline = line.strip().split('\t')
        # 使用map映射，将float()函数使用到curline中的每一个元素中，其实就是将这两个feature转化为浮点数的省略写法

        # 机器学习实战这本书有问题的，在使用map映射float到curline后没有将它转化为list，这回出现严重的问题！
        newline = list(map(float, curline))
        data.append(newline)
    return data


# 计算欧式距离
def distElud(vecA,vecB):
    return np.sqrt(np.sum(np.square(vecA - vecB), axis=1))


# 初始化聚类中心
def initializeCent1(dataset, k):
    # 求出每一个样本的特征数
    num_features = dataset.shape[1]
    center = np.mat(np.zeros(shape=(k, num_features)))
    for j in range(num_features):
        # 求出最大值和最小值之间的差距（范围）
        rangej = float(max(np.array(dataset)[:, j]) - min(dataset[:, j]))
        center[:, j] = np.random.rand(k, 1) * rangej + min(dataset[:, j])
    return center


def initializeCent2(dataset, k):
    center = []
    center.extend(dataset[random.sample(range(0, dataset.shape[0]), k)])

def K_means(dataset, k, dist=distElud, createCent=initializeCent1):
    num_samples = dataset.shape[0]
    clusterAssment = np.zeros(shape=(num_samples, 2))
    center = createCent(dataset, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged =False
        # 先取出第一个样本
        for i in range(num_samples):
            # 先初始化和聚类中心的欧式距离为正无穷大
            minDist = math.inf
            # 初始化后其对应的簇所属的聚类中心序号为-1
            minindex = -1
            for j in range(k):
                # 对每一个初始聚类中心求欧式距离，并记录最小欧式距离对应的距离和聚类中心的序号
                distIJ = dist(dataset[i, :], center[j, :])
                if distIJ < minDist:
                    minDist = distIJ
                    minindex = j
            # 当对一个样本完成所有聚类中心的测算之后

            # 若发现经过测算，出现了每个点的对应的聚类中心序号发生了变化（和之前对应的序号不一样了），则将clusterChanged标记为True，目的是将while循环继续下去，暂时还不能停下来
            if clusterAssment[i, 0] != minindex:
                clusterChanged = True
            # 在簇评测表中记录下该样本所属的聚类中心的序号以及欧式距离的平方
            clusterAssment[i, :] = minindex, minDist ** 2
        # d当第一轮样本全部划分完之后，输出最后的聚类中心点！
        print(center)
        for cent in range(k):
            # 找出所有的同类点，然后计算其均值，将其作为新的聚类中心！
            dataCent = dataset[np.nonzero(clusterAssment[:, 0] == cent)]
            center[cent, :] = np.mean(dataCent, axis=0)

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(num_samples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(center[i, 0], center[i, 1], mark[i], markersize=12)
    plt.show()

if __name__ == '__main__':
    dataSet = np.mat(loadData('./dataset/k-means/testSet.txt'))
    k = 3
    K_means(dataSet, k)




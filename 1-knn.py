import numpy as np

# 创建数据
def createdata():
    group = np.array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5]])
    labels = ['a', 'a', 'b', 'b']
    return group, labels

# 通过KNN进行分类
def classify(target, features, label, k):
    # 求出数据集的数据数目，方便对目标点进行扩增
    num_samples = features.shape[0]
    # 计算欧式距离
    diff = np.sqrt(np.sum(np.square((np.tile(target, (num_samples, 1)) - features)), axis=1))
    # 对距离进行排序-默认是返回由小到大的排列的元素的角标，若想要返回由大到小的，则在第一个参数前加上‘-’
    sorted_index = np.argsort(diff)

    classCount = {}

    # 对前k个邻居的种类数目进行统计
    for i in range(k):
        voteLabel = label[sorted_index[i]]
        # 使用dictionary的get的方法，第一个参数返回指定键对应的值，第二个参数给出找不到的情况下返回的值
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # 选出k个邻居多数人站在哪个类别里面
    max_count = 0
    chooseclass = ''
    for key, value in classCount.items():
        if value > max_count:
            max_count = value
            chooseclass = key
    return chooseclass


if __name__ == '__main__':
    features, labels = createdata()
    # 要预测的目标点
    target = np.array([1.1, 0.3])
    # 确定KNN的参数k值
    k = 3
    output = classify(target, features=features, label=labels, k=k)
    print("测试数据为：", target, "\n分类结果为：", output)

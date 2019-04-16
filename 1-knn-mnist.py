import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image


# 将01图象文件转化为1*1024的向量 32*32=1024
def img2vector(filename):
    # 写新的数组，建议在0数组中进行填充
    vec = np.zeros(shape=(1, 1024))
    file = open(filename)
    for i in range(32):
        # 每读一行
        linestr = file.readline()
        for j in range(32):
            # 一个一个字符进行写入
            vec[0, 32*i + j] = int(linestr[j])
    return vec

def classfy(target, features, labels, k):
    num_samples = features.shape[0]
    # 计算欧氏距离
    diff = np.sqrt(np.sum(np.square(np.tile(target, reps=(num_samples, 1)) - features), axis=1))
    sorted_index = np.argsort(diff)

    classCount = {}
    for i in range(k):
        # 得到排名靠前的标签名
        voteLabel = labels[sorted_index[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    max_class = 0
    for key, value in classCount.items():
        if value > max_class:
            max_class = value
            choose_class = key
    return choose_class

def handwrittingtest(k):
    hwLabels = []
    trainingFileList = os.listdir('./mnist/trainingDigits')
    num_trains = len(trainingFileList)
    trainingMat = np.zeros(shape=(num_trains, 1024))
    for i in range(num_trains):
        # 现在试着从文件名中取出标签,作为训练使用
        fileName = trainingFileList[i]
        classNum = fileName.split('.')[0].split('_')[0]
        # 存入标签集中
        hwLabels.append(classNum)
        # 开始将0 1 图变成vector
        trainingMat[i, :] = img2vector("./mnist/trainingDigits/%s" % fileName)  # trainingMat[i]

    # 制作测试集的features和labels
    testFileList = os.listdir('./mnist/testDigits')
    errorCount = 0
    num_test = len(testFileList)
    for i in range(num_test):
        # 开始逐个从test集中取出每一个样本进行预测，并标记正确与否，以计算准确率
        # 还是从文件名中取出其中的labels
        fileName = testFileList[i]
        classNum = fileName.split('_')[0]
        target = img2vector('./mnist/testDigits/%s' % fileName)
        result = classfy(target, features=trainingMat, labels=hwLabels, k=k)
        print("KNN分类的结果是：", result, "\t其真实结果是：", classNum)
        if result != classNum:
            errorCount += 1.0
    print("错误总数是：", errorCount)
    print("正确率是：", (num_test - errorCount)/num_test)
    return errorCount

if __name__ == '__main__':
    x = list()
    y = list()
    for i in range(1, 5):
        x.append(i)
        y.append(handwrittingtest(i))
    plt.plot(x, y)
    plt.show()


'''
    得出结论，当k=3的时候，效果最好！
'''

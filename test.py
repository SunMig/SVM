from svmMLiA import *

dataArr,labelArr=loadDataSet('testSet.txt')
# b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)
# #将得到的b,alpha输出
# print(b)
# print(alphas[alphas>0])#只输出大于0的量
# for i in range(100):
#     if alphas[i]>0:
#         print(dataArr[i],labelArr[i])
b,alphas=smoP(dataArr,labelArr,0.6,0.001,40)
print(b)
print(alphas)
#计算支持向量
ws=calcWs(alphas,dataArr,labelArr)
print(ws)

#测试某一个点属于哪一个分类,该值大于0属于1类，该值小于0属于-1类
dataMat=mat(dataArr)
print(dataMat[0])
type=dataMat[0]*mat(ws)+b
print(type)

#测试数据非线性可分时利用核函数的训练
testRbf()
#手写数字识别测试
# testDigits(('rbf',20))


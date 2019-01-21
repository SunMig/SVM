import matplotlib.pyplot as plt
from numpy import *
def loadDataSet(filename):
    dataMat=[];
    labelMat=[];
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
#函数的i是第一个alpha的下标，m是所有的alpha的数目，当函数值不等于输入值i时就随机选择
def selectJrand(i,m):
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j
#用于调整大于H或者是小于H的alphade值
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

#简版的SMO算法
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=mat(dataMatIn)
    labelMatrix=mat(classLabels).transpose()
    b=0;m,n=shape(dataMatrix)
    alphas=mat(zeros((m,1)))
    iter=0
    #循环maxIter次
    while (iter<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fxi=float(multiply(alphas,labelMatrix).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei=fxi-float(labelMatrix[i])
            #如果第一个alpha可以优化，进入优化过程
            if((labelMatrix[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMatrix[i]*Ei>toler) and (alphas[i]>0)):
                #随机选择另外一个alpha
                j=selectJrand(i,m)
                fxj=float(multiply(alphas,labelMatrix).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fxj-float(labelMatrix[j])
                alphaIold=alphas[i].copy();
                alphaJold=alphas[j].copy();
                #L和H用于将alpha[i]调整到0和C之间，保证alpha在0和C之间
                if(labelMatrix[i]!=labelMatrix[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                #如果二者相等，说明不用做任何修改，直接进行下一次循环
                if L==H: print("L==H");continue;
                #eta是alpha[j]的最优修改量
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                #如果eta为0退出本次循环
                if eta>=0:print("eta>=0");continue;
                #计算出一个alphas[j]
                alphas[j]-=labelMatrix[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                #检查alpha[j]是否有轻微的改变，这里是与之前的Jold作比较，如果是则退出本次循环
                if(abs(alphas[j]-alphaJold)<0.00001):print("j not moving enough");continue;
                #上述判断结束之后，同时对alpha[i]和alpha[j]进行改变修正，改变的大小一样但是方向相反（一个增大另外一个减少）
                alphas[i]+=labelMatrix[j]*labelMatrix[i]*(alphaJold-alphas[j])
                #修正完成之后，对两个alpha设置一个常数项
                b1=b-Ei-labelMatrix[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMatrix[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMatrix[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMatrix[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if(0<alphas[i]) and (C>alphas[i]):
                    b=b1
                elif(0<alphas[j])and(C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                #如果执行到for循环的最后没有执行依据contuniue语句，说明已经成功改变了一对alpha
                alphaPairsChanged+=1
                print("iter: "+str(iter)+" i: "+str(i)+" paris changed "+str(alphaPairsChanged))
        #在for循环之外判断alpha是否做了更新（iter来指示）这里alphaParisChanged一般不是0，代表更新了，然后把iter重置为0
        if(alphaPairsChanged==0):
            iter+=1
        else:
            iter=0
        print("iteration number: "+str(iter))
    return b,alphas

#完整版的Platt SMO函数
class optStruct:
    def __init__(self,dataMatIn,ClassLabels,C,toler,kTup):
        self.X=dataMatIn
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zeros((self.m,2)))
        self.labelMat=ClassLabels
        self.K=mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)
#计算误差的函数
def calEk(oS,k):
    #对某一行数据的属性进行预测
    #fXk=float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+oS.b
    fXk = float(multiply(oS.alphas, oS.labelMat).T *oS.K[:,k] + oS.b)
    #求误差
    Ek=fXk-float(oS.labelMat[k])
    return Ek
#选择第二个alpha值，这里保证每次优化采用最大化的步长，
# 简化版的算法是随机选择的，这是二者的区别所在
def selectJ(i,oS,Ei):
    maxK=-1;maxDeltaE=0;Ej=0;
    oS.eCache[i]=[1,Ei]
    validEcacheList=nonzero(oS.eCache[:,0].A)[0]
    if(len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i:
                continue;
            Ek=calEk(oS,k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxDeltaE):
                maxK=k;maxDeltaE=deltaE;Ej=Ek;
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calEk(oS,j)
        return j,Ej
#将计算的误差值存入缓存中，再对alpha值优化之后会用到这个值
def updateEk(oS,k):
    Ek=calEk(oS,k)
    oS.eCache[k]=[1,Ek]

#寻找决策边界的优化例程,选择第二个alpha值
def innerL(i,oS):
    Ei=calEk(oS,i);
    if((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
        j,Ej=selectJ(i,oS,Ei)
        alphaIold=oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:print(" L==H ");return 0;
        #eta是最优的修改量
        #eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T #线性可分
        eta=2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j] #线性不可分时，使用核函数
        if eta>=0:print("eta>=0");return 0;
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        #更新误差值
        updateEk(oS,j)
        if(abs(oS.alphas[j]-alphaJold)<0.0001):
            print("j not moving enough ")
            return 0
        oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)
        #b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i,j]
        #b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[j, j]
        if(0<oS.alphas[i]) and (oS.C>oS.alphas[i]):
            oS.b=b1
        elif(0<oS.alphas[j]) and (oS.C>oS.alphas[j]):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2
        return 1
    else:
        return 0

#外循环，选择第一个alpha值
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS=optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter=0
    entireSet=True;alphaParisChanged=0
    while (iter<maxIter) and ((alphaParisChanged>0) or (entireSet)):
        alphaParisChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaParisChanged+=innerL(i,oS)
                print("fullSet,iter: "+str(iter)+" i: "+str(i)+" pairs changed "+str(alphaParisChanged))
            iter=iter+1
        else:
            nonBoundIs=nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaParisChanged+=innerL(i,oS)
                print("non-bound,iter: "+str(iter)+" i: "+str(i)+" pairs changed "+str(alphaParisChanged))
            iter+=1
        if entireSet:
            entireSet=False
        elif(alphaParisChanged==0):
            entireSet=True
            print("iteration number: "+str(iter))
    #返回常数项b以及alphas的数组
    return oS.b,oS.alphas

#超平面法向量的计算方法
def calcWs(alphas,dataArr,classLabels):
    X=mat(dataArr)
    labelMat=mat(classLabels).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    for i in range(m):
        w+=multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

#核转换函数，当数据不是线性可分时应用核函数进行映射
def kernelTrans(X,A,kTup):
    m,n=shape(X)
    K=mat(zeros((m,1)))
    if kTup[0]=='lin':
        K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Houston We have a problem -- That Kernel is not recognized')
    return K

#测试中使用核函数
def testRbf(k1=1.3):
    dataArr,labelArr=loadDataSet('testSetRBF.txt')
    b,alphas=smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat=mat(dataArr);labelMat=mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]#获取支持向量
    sVs=dataMat[svInd]
    print(sVs)
    labelSV=labelMat[svInd]
    print("there are "+str(shape(sVs)[0])+" Support Vectors")
    print(sVs)
    m,n=shape(dataMat)
    errorCount=0;
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if(sign(predict)!=sign(labelArr[i])):
            errorCount+=1
    print("the training error rate is "+str(float(errorCount/m)))
    #测试数据集，检验下分类器的精度
    dataArr,labelArr=loadDataSet('testSetRBF2.txt')
    errorCount=0
    dataMat=mat(dataArr);labelMat=mat(labelArr).transpose()
    xcord0=[];ycord0=[];xcord1=[];ycord1=[];
    m,n=shape(dataMat)
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):
            errorCount+=1
        #记录下坐标值
        if(sign(predict)==1):
            xcord0.extend(dataMat[i,:].tolist()) #1类的点
        else:
            xcord1.extend(dataMat[i,:].tolist()) #-1类的点
    #展出散点图
    fig=plt.figure();
    ax=fig.add_subplot(111)
    s1=shape(xcord0)[0]
    for i in range(s1):
        ax.scatter(xcord0[i][0],xcord0[i][1],marker='o',s=50,c='blue')
    s1=shape(xcord1)[0]
    for i in range(s1):
        ax.scatter(xcord1[i][0],xcord1[i][1],marker='s',s=50,c='red')
    s1=len(sVs)
    for i in range(s1):
        ycord1.extend(sVs[i,:].tolist())
    for i in range(s1):
        ax.scatter(ycord1[i][0],ycord1[i][1], marker='o', edgecolors='y',s=200,color='')
    print("the test data error rate is " + str(float(errorCount / m)))
    plt.show()









#手写数字识别
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print ("there are %d Support Vectors"+str(shape(sVs)[0]))
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print ("the training error rate is: "+str((float(errorCount)/m)))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print ("the test error rate is: "+str((float(errorCount)/m)))
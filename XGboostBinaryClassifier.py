#coding=utf-8

import xgboost
import time
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
import warnings
#warnings.filterwarnings("ignore")

class XGBoostClassifier:

    def initParameters(self):
        self.params={
            'booster':'gbtree',
            'objective':'binary:logistic', #多分类的问题
            'n_estimators':100,
            'learning_rate':0.05,
            'gamma':0.0,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth':6, # 构建树的深度，越大越容易过拟合
            'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'subsample':0.6, # 随机采样训练样本
            'colsample_bytree':0.6, # 生成树时进行的列采样
            'min_child_weight':1,
            #这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言,
            #假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            #'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
            'eta': 0.007, # 如同学习率
            'seed':1000,
            'reg_alpha':1e-5,
            'verbose':0,
            'n_jobs':-1,
            'silent':1
            }

    def updateParameters(self,new_paras):
        for k in new_paras:
            self.params[k]=new_paras[k]


    def __init__(self):

        self.trainEpchos=500
        self.threshold=0.5
        self.initParameters()
        self.name="XGBoostClassifier"

    def predict(self,X):

        print(self.name,"XGBoost model is predicting")

        Y=self.model.predict(X)

        return Y

    def navieTrain(self,dataSet):

        print(" navie training")
        t0=time.time()
        trainX,trainY=dataSet.trainX,dataSet.trainY

        #begin to search best parameters
        self.model=xgboost.XGBClassifier(params=self.params)
        self.model.fit(trainX,trainY,eval_metric=metrics.make_scorer(metrics.f1_score))
        t1=time.time()
        #measure training result
        vpredict=self.predict(trainX)
        #print(vpredict[:3])
        score=metrics.f1_score(trainY,vpredict)
        print("model",self.name,"trainning finished in %ds"%(t1-t0),"validate score=%f"%score)

    def searchParameters(self,dataSet):
        print(" search training")
        t0=time.time()

        paraSelection=[
            {'n_estimators':[i for i in range(100,500,50)],'learning_rate':[i/100.0 for i in range(5,30,5)]},
            {'max_depth':[i for i in range(6,15)],'min_child_weight':[i for i in range(1,6)]},
            {'gamma':[i/10.0 for i in range(0,5)]},
            {'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]},
            {'reg_alpha':[1e-5, 1e-2, 0.1, 1.0, 100.0]},
        ]

        for i in range(len(paraSelection)):

            para1=paraSelection[i]

            self.model=xgboost.XGBClassifier(**self.params)
            gsearch=GridSearchCV(self.model,para1,verbose=0,scoring=metrics.make_scorer(metrics.f1_score))
            gsearch.fit(dataSet.trainX,dataSet.trainY)
            print("best paras",gsearch.best_params_)
            self.updateParameters(gsearch.best_params_)

        print("para search finished in %ds"%(time.time()-t0))


    def trainModel(self,dataSet):

        #procedure 1=>search best parameters

        self.searchParameters(dataSet)

        #procedure 2=> train true model
        self.navieTrain(dataSet)

    def saveModel(self):
        modelpath="./models/"+self.name+".pkl"
        with open(modelpath,"wb") as f:
            pickle.dump(self.model,f)

    def loadModel(self):
        modelpath = "./models/" + self.name + ".pkl"
        with open(modelpath, "rb") as f:
            self.model=pickle.load(f)

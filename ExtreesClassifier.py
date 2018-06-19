from sklearn import ensemble
from sklearn import metrics
import time,pickle
from sklearn.model_selection import GridSearchCV
import warnings
import numpy as np

warnings.filterwarnings("ignore")

class TreeClassifier:

    def initParameters(self):
        self.params={
            'n_estimators':60,
            'criterion':"gini",
            'max_depth':12,
            'min_samples_split':25,
            'min_samples_leaf':13,
            'max_features':"auto",
            'max_leaf_nodes':None,
            'bootstrap':False,
            'n_jobs':-1,
            'verbose':0,
            'class_weight':{0:1,1:1}
        }

    def __init__(self):
        self.name="ExtreesClassifier"
        self.initParameters()

    def predict(self,X):

        Y=self.model.predict(X)
        return Y

    def updateParameters(self,paras):
        for k in paras:
            self.params[k]=paras[k]

    def searchParameters(self,dataSet):
        print("searching for best parameters")

        selParas=[
            {'n_estimators':[i for i in range(10,200,10)]},
            {'criterion':["gini","entropy"]},
            {'max_depth':[i for i in range(3,20)]},
            {'min_samples_split':[i for i in range(20,100,5)]},
            {'min_samples_leaf':[i for i in range(5,30,2)]},
            {'max_features':["auto","sqrt","log2",None]},
            {'class_weight':[{0:1,1:i} for i in range(1,7)]}
        ]


        for i in range(len(selParas)):
            para=selParas[i]
            model=ensemble.ExtraTreesClassifier(**self.params)
            gsearch=GridSearchCV(model,para,scoring=metrics.make_scorer(metrics.f1_score))
            gsearch.fit(dataSet.trainX,dataSet.trainY)
            print("best para",gsearch.best_params_)
            self.updateParameters(gsearch.best_params_)

        self.model=ensemble.ExtraTreesClassifier(**self.params)

    def trainModel(self,dataSet):
        print("training")
        t0=time.time()

        self.searchParameters(dataSet)

        self.model.fit(dataSet.trainX,dataSet.trainY)

        t1=time.time()

        #measure training result
        vpredict=self.predict(dataSet.trainX)
        #print(vpredict)
        score=metrics.f1_score(dataSet.trainY,vpredict)
        cm=metrics.confusion_matrix(dataSet.trainY,vpredict)
        print("model",self.name,"trainning finished in %ds"%(t1-t0),"validate score=%f"%score,"CM=\n",cm)

    def saveModel(self):
        modelpath="./models/"+self.name+".pkl"
        with open(modelpath,"wb") as f:
            pickle.dump(self.model,f)

    def loadModel(self):
        modelpath = "./models/" + self.name + ".pkl"
        with open(modelpath, "rb") as f:
            self.model=pickle.load(f)

from keras import models,layers,optimizers,losses
import time,keras.backend as K
from keras import utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
import warnings,pickle
import numpy as np
import collections
from initConfig import config
warnings.filterwarnings("ignore")

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



#create model
def createDNN(dp=0.8):
    inputDim=2*config["features"]
    ouputDim=2
    DNNmodel=models.Sequential()
    DNNmodel.add(layers.Dense(units=160,input_shape=(inputDim,),activation="relu"))
    DNNmodel.add(layers.Dense(units=128,activation="relu"))
    DNNmodel.add(layers.Dense(units=128, activation="relu"))
    DNNmodel.add(layers.Dense(units=128))
    DNNmodel.add(layers.Dropout(dp))
    DNNmodel.add(layers.Dense(units=ouputDim))

    opt = optimizers.Adadelta()
    DNNmodel.compile(optimizer=opt,loss=losses.binary_crossentropy)
    return DNNmodel

#model
class DNNCLassifier:

    def __init__(self):
        self.name="DNNBinaryClassifier"
        self.params={

            'dp':0.8,
            'verbose':0,
        }


    def trainModel(self,dataSet):
        print(self.name+" training")
        t0=time.time()
        counter=collections.Counter(dataSet.trainY)
        l1=float(counter[0])
        l2=float(counter[1])
        l=max(l1,l2)
        cls_w={0:100*l2/l,1:100*l1/l}
        print(counter)
        print("class weight",cls_w)
        self.model=createDNN(self.params["dp"])

        self.model.fit(dataSet.trainX,utils.to_categorical(dataSet.trainY,2),verbose=2,epochs=2000,batch_size=1000,class_weight=cls_w)

        t1=time.time()
        y_predict=self.predict(dataSet.trainX)

        f1=metrics.f1_score(dataSet.trainY,y_predict)
        acc=metrics.accuracy_score(dataSet.trainY,y_predict)
        print("finished in %ds"%(t1-t0),"f1=",f1,"acc=",acc)

    def predict(self,X):

        Y=self.model.predict(X,verbose=0)
        Y=np.argmax(Y,axis=1)
        return Y

    def loadModel(self):

        self.model = models.load_model("./models/" + self.name + ".h5")

        print("loaded",self.name,"model")

    def saveModel(self):
        self.model.save("./models/" + self.name + ".h5")
        print("saved",self.name,"model")
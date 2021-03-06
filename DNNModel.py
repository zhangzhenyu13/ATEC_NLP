from keras import models,layers,optimizers,losses,metrics
import time
from keras import utils
import warnings
import numpy as np
import initConfig
from sklearn import metrics
from keras.callbacks import Callback
import keras.backend as K
from FeatureDataSet import NLPDataSet
warnings.filterwarnings("ignore")

class MyMetrics(Callback):
    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_cms=[]
        self.val_acc=[]
    def on_epoch_end(self, epoch, logs=None):

        '''
        print(type(self.validation_data))
        for i in range(len(self.validation_data)):
            print(np.shape(self.validation_data[i]))

        print("show data")
        for i in range(2,len(self.validation_data)):
            print(self.validation_data[i][:10])
        '''


        feed_data = {"em1": self.validation_data[0], "em2": self.validation_data[1]}
        # feed_label={"label":self.validation_data[2],"Y":self.validation_data[3]}
        predicts=self.model.predict(feed_data)#[0]
        #print(predicts)
        val_predict = np.argmax(predicts,axis=1)
        val_targ = np.argmax(self.validation_data[2],axis=1)

        #metrics print
        CM=metrics.confusion_matrix(val_targ,val_predict)
        tn, fp, fn, tp = CM.ravel()
        if tp+fp==0:
            precision=0
        else:
            precision = float(tp) / float(tp + fp)
        if fn+tp==0:
            recall=0
        else:
            recall = float(tp) / float(fn + tp)

        print("tp,fp,tn,fn", tp, fp, tn, fn)
        print(CM)
        print("precision,recall", precision, recall)
        # measure
        f1 = 2 * precision * recall / (precision + recall)

        print("f1-score=", f1)

        acc = float(tp + tn) / float(tn + fp + fn + tp)

        print("accuracy=", acc)

        print()
        return

#model
class TwoInDNNModel:

    def __init__(self):
        self.name=None
        self.model=None
        self.numEpoch=20
    def buildModel(self):
        '''
        define deep learning net
        :return:
        '''
        pass


    def trainModel(self,train,validate):
        from keras.callbacks import TensorBoard
        import collections
        print(collections.Counter(train.dataY))
        tensorboard = TensorBoard(log_dir="/tmp/zhangzyTFK",histogram_freq=1)
        watch_metrics = MyMetrics()

        print(self.name+" training")
        t0=time.time()

        cls_w={0:1,1:4.45}

        print("class weight",cls_w)

        self.buildModel()

        em1,em2 = train.dataEm1,train.dataEm2
        label = train.dataY
        feeddata={"em1":em1,"em2":em2}
        feedlabel={
            "label":utils.to_categorical(label,2)
        }

        if validate is not None:

            val_em1, val_em2 = validate.dataEm1, validate.dataEm2
            val_label = validate.dataY
            val_feeddata = {"em1": val_em1, "em2": val_em2}
            val_feedlabel = {
                "label": utils.to_categorical(val_label, 2)
            }
            val_data=(val_feeddata,val_feedlabel)
        else:

            val_data=None

        self.model.fit(feeddata,feedlabel,
            verbose=2, epochs=self.numEpoch, batch_size=500

                       ,validation_data=val_data
                       ,class_weight={"label":cls_w}
                       ,callbacks=[watch_metrics]
                       #,validation_split=0.2
                       )

        t1=time.time()

        print("finished in %ds"%(t1-t0))

        #print(self.model.get_weights())

    def predict(self,dataSet):
        em1, em2 = dataSet.dataEm1, dataSet.dataEm2
        feeddata = {"em1": em1, "em2": em2}

        Y=self.model.predict(feeddata,verbose=0)
        #print(Y)
        #Y=Y[0]
        Y=np.argmax(Y,axis=1)

        print(self.name,"finished predicting %d records"%len(em1))
        return Y

    def loadModel(self):
        self.buildModel()
        self.model.load_weights("./models/" + self.name + ".h5")

        print("loaded",self.name,"model")

    def saveModel(self):
        self.model.save_weights("./models/" + self.name + ".h5")

        print("saved",self.name,"model")

#ini func
def W_init(shape, name=None):
    """Initialize weights as in paper"""
    values = np.random.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)

def b_init(shape, name=None):
    """Initialize bias as in paper"""
    values = np.random.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def getFeedData(dataPath,emModel):
    data = NLPDataSet(testMode=False)
    data.loadDocsData(dataPath)
    docs = data.getAllDocs()

    embeddings = emModel.transformDoc2Vec(docs)

    n_count = len(embeddings)
    em1 = embeddings[:n_count // 2]
    em2 = embeddings[n_count // 2:]

    labels = np.array(data.docdata["label"], dtype=np.int)

    data.constructData(em1=em1, em2=em2, labels=labels)

    return data

def getFeedDataInit(data,emModel):
    docs = data.getAllDocs()

    embeddings = emModel.transformDoc2Vec(docs)

    n_count = len(embeddings)
    em1 = embeddings[:n_count // 2]
    em2 = embeddings[n_count // 2:]

    labels = np.array(data.docdata["label"], dtype=np.int)

    data.constructData(em1=em1, em2=em2, labels=labels)

    return data

def trainModel(MyModel,paraName="lstmNum"):
    from WordModel import WordEmbedding
    from FeatureDataSet import NLPDataSet
    from utilityFiles import splitTrainValidate
    from imblearn import over_sampling
    # embedding words
    emModel = WordEmbedding()
    emModel.loadModel()

    splitratio = 0.8
    if splitratio > 0 and splitratio < 1:
        splitTrainValidate("../data/train_nlp_data.csv", splitratio)

        trainData = getFeedData("../data/train.csv", emModel)

        # resample methods
        '''
        datashape = (initConfig.config["maxWords"], initConfig.config["features"])

        X,Y=np.concatenate((trainData.dataEm1,trainData.dataEm2),axis=1),trainData.dataY
        X=np.reshape(X,newshape=(len(X),2*datashape[0]*datashape[1]))
        osam=over_sampling.ADASYN(n_jobs=10)
        X,Y=osam.fit_sample(X,Y)

        trainData.dataEm1,trainData.dataEm2,trainData.dataY=\
            np.reshape(X[:,:datashape[0]*datashape[1]],newshape=(len(X),datashape[0],datashape[1])),\
            np.reshape(X[:,datashape[0]*datashape[1]:],newshape=(len(X),datashape[0],datashape[1])),\
            Y
        '''

        validateData = getFeedData("../data/validate.csv", emModel)
        dnnmodel = MyModel()

        dnnmodel.trainModel(trainData, validateData)
        dnnmodel.saveModel()
        exit(1)
    else:
        trainData, validateData = getFeedData("../data/train_nlp_data.csv", emModel), None

    model_num = initConfig.config[paraName]
    dataList = trainData.getFold(model_num)
    for i in range(model_num):
        # lstm dnn model
        dnnmodel = MyModel()

        dnnmodel.name += str(i)
        train, test = dataList[i]
        dnnmodel.trainModel(train, test)
        dnnmodel.saveModel()

        print("\n==========%d/%d=================\n" % (i + 1, model_num))
    exit(2)

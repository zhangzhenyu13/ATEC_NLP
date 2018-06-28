from keras import models,layers,optimizers,losses,metrics
import time
from keras import utils
import warnings
import numpy as np
import initConfig
from sklearn import metrics
from keras.callbacks import Callback
warnings.filterwarnings("ignore")

class Metrics(Callback):
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
        predicts=self.model.predict(feed_data)
        #print(predicts)
        val_predict = np.argmax(predicts[0],axis=1)
        val_targ = np.reshape(np.array(self.validation_data[3],dtype=np.int),newshape=len(val_predict))

        #metrics print
        CM=metrics.confusion_matrix(val_targ,val_predict)
        tn, fp, fn, tp = CM.ravel()
        precision = float(tp) / float(tp + fp)
        recall = float(tp) / float(fn + tp)
        print("tp,fp,tn,fn", tp, fp, tn, fn)
        print(CM)
        print("precision,recall", precision, recall)
        # measure
        f1 = 2 * precision * recall / (precision + recall)

        print("f1-score=", f1)

        acc = float(tp + tn) / float(tn + fp + fn + tp)

        print("accuracy=", acc)

        return

#model
class TwoInDNNModel:

    def __init__(self):
        self.name="TwoInputDNN"

    def buildModel(self):
        datashape=(initConfig.config["maxWords"],initConfig.config["features"])

        #word net
        input1=layers.Input(shape=datashape,name="em1")
        input2=layers.Input(shape=datashape,name="em2")
        comLSTM=layers.LSTM(128)
        encode1=comLSTM(input1)
        encode2=comLSTM(input2)
        features=layers.concatenate([encode1,encode2],axis=1)

        # extract net2

        hiddenLayer=layers.Dense(units=196,activation="relu")(features)
        predictionLayer=layers.Dense(units=2,activation="sigmoid",name="label")(hiddenLayer)
        regLayer=layers.Dense(units=1,activation="sigmoid",name="Y")(hiddenLayer)
        self.model=models.Model(inputs=[input1,input2],outputs=[predictionLayer,regLayer])

        self.model.compile(optimizer="rmsprop",
                      loss={"label":losses.binary_crossentropy,"Y":losses.mse}
                      )

        return self.model


    def trainModel(self,train,validate):
        from keras.callbacks import TensorBoard

        tensorboard = TensorBoard(log_dir="/tmp/zhangzyTFK",histogram_freq=1)
        watch_metrics = Metrics()

        print(self.name+" training")
        t0=time.time()

        cls_w={0:1,1:2}
        print("class weight",cls_w)

        '''
        sample_weight=np.zeros(shape=len(dataSet.dataY))
        for i in range(len(sample_weight)):
            if dataSet.dataY[i]==0:
                sample_weight[i]=cls_w[0]
            else:
                sample_weight[i]=cls_w[1]
        '''


        self.buildModel()

        em1,em2 = train.dataEm1,train.dataEm2
        label = train.dataY
        feeddata={"em1":em1,"em2":em2}
        feedlabel={
            "label":utils.to_categorical(label,2)
            ,"Y":label
        }

        if validate is not None:

            val_em1, val_em2 = validate.dataEm1, validate.dataEm2
            val_label = validate.dataY
            val_feeddata = {"em1": val_em1, "em2": val_em2}
            val_feedlabel = {
                "label": utils.to_categorical(val_label, 2)
                , "Y": val_label
            }
            val_data=(val_feeddata,val_feedlabel)
        else:

            val_data=None

        self.model.fit(feeddata,feedlabel,
            verbose=2, epochs=30, batch_size=500
                       #,sample_weight={"Y":sample_weight}
                       ,validation_data=val_data
                       ,class_weight={"label":cls_w}
                       ,callbacks=[tensorboard,watch_metrics]
                       ,validation_split=0.2
                       )

        t1=time.time()

        print("finished in %ds"%(t1-t0))

    def predict(self,dataSet):
        em1, em2 = dataSet.dataEm1, dataSet.dataEm2
        feeddata = {"em1": em1, "em2": em2}

        Y=self.model.predict(feeddata,verbose=0)
        #print(Y)
        Y=Y[0]
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


def getFeedData(dataPath):
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

if __name__ == '__main__':
    from WordModel import WordEmbedding
    from FeatureDataSet import NLPDataSet
    from utilityFiles import splitTrainValidate

    # embedding words
    emModel = WordEmbedding()
    emModel.loadModel()

    #lstm dnn model
    dnnmodel=TwoInDNNModel()

    splitratio=0.8
    if splitratio>0 and splitratio<1:
        splitTrainValidate("../data/train_nlp_data.csv",splitratio)

        trainData=getFeedData("../data/train.csv")
        validateData=getFeedData("../data/validate.csv")
    else:
        trainData,validateData=getFeedData("../data/train_nlp_data.csv"),None

    dnnmodel.trainModel(trainData,validateData)
    dnnmodel.saveModel()
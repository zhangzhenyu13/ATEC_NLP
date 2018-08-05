from keras import models,layers,optimizers,losses,metrics
import time
from keras import utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import warnings
import numpy as np
import pandas as pd
import initConfig
from sklearn import metrics
from keras.callbacks import Callback
import keras.backend as K
import pickle
warnings.filterwarnings("ignore")

class NLPDataSet:
    def __init__(self,testMode):

        self.docdata=None
        self.testMode=testMode

    def loadDocsData(self,inputfile):

        with open(inputfile,"r") as f:
            records=[]
            for line in f:
                record=line.replace("\n","").replace("\r", "").split("\t")
                records.append(record)
        if self.testMode:
            self.docdata=pd.DataFrame(data=records,columns=["id","sent1","sent2"])
        else:
            self.docdata=pd.DataFrame(data=records,columns=["id","sent1","sent2","label"])

        print("loaded %d records"%len(self.docdata["id"]))


    def constructData(self):
        """

        :param s1:
        :param s2:
        :param labels:
        :return:
        """
        s1, s2 = self.docdata["sent1"], self.docdata["sent2"]
        self.text1=s1
        self.text2=s2
        if self.testMode:
            self.label=np.zeros(shape=len(s1),dtype=np.int)
        else:
            self.label = np.array(self.docdata["label"],dtype=np.int)

    def getFold(self,fold_num=5):
        from sklearn import model_selection
        kfold=model_selection.KFold(n_splits=fold_num,shuffle=True)
        dataList=[]
        Y=np.reshape(self.label,newshape=(len(self.label),1))

        for train_index, test_index in kfold.split(Y):

            data_train=NLPDataSet(False)
            data_test=NLPDataSet(False)

            data_train.text1,data_train.text2,data_train.label=\
                self.text1[train_index],self.text2[train_index],self.label[train_index]
            data_test.text1,data_test.text2,data_test.label=\
                self.text1[test_index],self.text2[test_index],self.label[test_index]

            dataList.append([data_train,data_test])

        return dataList


    def getInitialFold(self,fold_num=5):
        from sklearn import model_selection
        kfold = model_selection.KFold(n_splits=fold_num, shuffle=True)
        dataList = []
        Y=np.array(self.docdata["label"])
        Y = np.reshape(Y, newshape=(len(Y), 1))
        for train_index, test_index in kfold.split(Y):
            data_train=NLPDataSet(False)
            data_test=NLPDataSet(False)
            data_train.docdata=self.docdata.loc[train_index]
            data_test.docdata=self.docdata.loc[test_index]

            dataList.append([data_train,data_test])

        return dataList

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


        feed_data = {"seq1": self.validation_data[0], "seq2": self.validation_data[1]}
        predicts=self.model.predict(feed_data)#[0]
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
def buildTokenizer(tokenModel,data):
        #tkonizer model
        t0=time.time()
        text1,text2=data.text1,data.text2
        texts=np.concatenate((text1,text2),axis=0)
        tokenModel.fit_on_texts(texts)
        t1=time.time()
        print("tokenizer building finished in %ds"%(t1-t0))

class NlpModel:
    tokenModel=Tokenizer(num_words=10000)
    def __init__(self):
        self.name="NlpDecisionModel"
        self.model=None
        self.numEpoch=20
        self.maxLen=256
        self.embeddingSize=256

        self.model_dir="./models/"

    def buildModel(self):
        datashape=(initConfig.config["maxWords"],initConfig.config["features"])

        #word net
        input1=layers.Input(shape=datashape,name="seq1")
        input2=layers.Input(shape=datashape,name="seq2")

        comEmbedding=layers.Embedding()
        comLSTM=layers.Bidirectional(layers.LSTM(128,return_sequences=True))
        encode1=comLSTM(input1)
        encode2=comLSTM(input2)

        comFlat=layers.Flatten()
        encode1=comFlat(encode1)
        encode2=comFlat(encode2)
        # extract net2

        L1_distance = lambda x: K.abs(x[0] - x[1])
        both = layers.merge([encode1, encode2], mode=L1_distance, output_shape=lambda x: x[0])

        hiddenLayer=layers.Dense(units=256,activation="relu")(both)
        dropLayer=layers.Dropout(0.36)(hiddenLayer)
        predictionLayer=layers.Dense(units=2,name="label",activation="softmax")(dropLayer)
        self.model=models.Model(inputs=[input1,input2],
                                outputs=[
                                    predictionLayer,
                                ]
                                )

        self.model.compile(optimizer=optimizers.Adam(),
                      loss={
                          "label":losses.binary_crossentropy
                           }
                      )

        return self.model


    def trainModel(self,train,validate):
        from keras.callbacks import TensorBoard
        import collections
        print(collections.Counter(train.dataY))
        tensorboard = TensorBoard(log_dir="/tmp/zhangzyTFK",histogram_freq=1)
        watch_metrics = MyMetrics()

        print(self.name+" training")

        cls_w={0:1,1:4.45}

        print("class weight",cls_w)


        #siamese networks
        t0=time.time()

        self.buildModel()

        label = train.label
        text1,text2=train.text1,train.text2
        seq1=self.tokenModel.texts_to_sequences(text1)
        seq2=self.tokenModel.texts_to_sequences(text2)
        seq1,seq2=sequence.pad_sequences(seq1,self.maxLen),sequence.pad_sequences(seq2,self.maxLen)
        feeddata={"seq1":seq1,"seq2":seq2}
        feedlabel={
            "label":utils.to_categorical(label,2)
        }

        if validate is not None:

            vtext1,vtext2=validate.text1,validate.text2
            vseq1,vseq2=self.tokenModel.texts_to_sequences(vtext1),self.tokenModel.texts_to_sequences(vtext2)
            vseq1,vseq2=sequence.pad_sequences(vseq1,self.maxLen),sequence.pad_sequences(vseq2,self.maxLen)
            val_label = validate.label
            val_feeddata = {"seq1":vseq1,"seq2":vseq2}
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

        print("siamese model(%s) training finished in %ds"%(self.name,t1-t0))

        #print(self.model.get_weights())

    def predict(self,dataSet):
        text1,text2=dataSet.text1, dataSet.text2
        seq1=self.tokenModel.texts_to_sequences(text1)
        seq2=self.tokenModel.texts_to_sequences(text2)
        seq1,seq2=sequence.pad_sequences(seq1,self.maxLen),sequence.pad_sequences(seq2,self.maxLen)
        feeddata = {"seq1": seq1, "seq2": seq2}

        Y=self.model.predict(feeddata,verbose=0)

        Y=np.argmax(Y,axis=1)

        print(self.name,"finished predicting %d records"%len(text1))
        return Y

    def loadModel(self):
        self.model=models.load_model(self.model_dir + self.name + ".h5")
        with open(self.model_dir+"tokneModel","rb") as f:
            self.tokenModel=pickle.load(f)

        print("loaded",self.name,"model")

    def saveModel(self):
        self.model.save(self.model_dir + self.name + ".h5")
        with open(self.model_dir+"tokneModel","wb") as f:
            pickle.dump(self.tokenModel,f)

        print("saved",self.name,"model")


def trainModel(MyModel):

    data=NLPDataSet(False)
    data.loadDocsData("./trainData/train_nlp_data.csv")
    data.constructData()
    buildTokenizer(MyModel.tokenModel,data)
    dataList=data.getFold(model_num)

    for i in range(model_num):
        # lstm dnn model
        dnnmodel = MyModel()

        dnnmodel.name += str(i)
        train, test = dataList[i]
        dnnmodel.trainModel(train, test)
        dnnmodel.saveModel()

        print("\n==========%d/%d=================\n" % (i + 1, model_num))

if __name__ == '__main__':
    model_num = 5
    trainModel(NlpModel)

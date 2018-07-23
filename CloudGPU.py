import pandas as pd
from keras import models,layers,optimizers,losses,metrics
import time
from keras import utils
import warnings
import numpy as np
from sklearn import metrics
from keras.callbacks import Callback
import keras.backend as K
from sklearn import model_selection
import json
warnings.filterwarnings("ignore")

#config
maxWords=30
features=128
model_num = 10
#
class NLPDataSet:
    def __init__(self):
        self.docdata=None

    def loadDocsData(self,df):

        self.docdata=df

        print("loaded %d records"%len(self.docdata["id"]))

    def constructData(self,em1,em2,labels):
        """
        :param s1:
        :param s2:
        :param labels:
        :return:
        """

        self.dataEm1=em1
        self.dataEm2=em2
        self.dataY = labels

    def getFold(self,fold_num=5):
        kfold=model_selection.KFold(n_splits=fold_num,shuffle=True)
        dataList=[]
        Y=np.reshape(self.dataY,newshape=(len(self.dataY),1))
        for train_index, test_index in kfold.split(Y):
            data_train=NLPDataSet()
            data_test=NLPDataSet()
            data_train.dataEm1,data_train.dataEm2,data_train.dataY=\
                self.dataEm1[train_index],self.dataEm2[train_index],self.dataY[train_index]
            data_test.dataEm1,data_test.dataEm2,data_test.dataY=\
                self.dataEm1[test_index],self.dataEm2[test_index],self.dataY[test_index]

            dataList.append([data_train,data_test])

        return dataList

    def getInitialFold(self,fold_num=5):
        from sklearn import model_selection
        kfold = model_selection.KFold(n_splits=fold_num, shuffle=True)
        dataList = []
        Y=np.array(self.docdata["label"])
        Y = np.reshape(Y, newshape=(len(Y), 1))
        for train_index, test_index in kfold.split(Y):
            data_train=NLPDataSet()
            data_test=NLPDataSet()
            data_train.docdata=self.docdata.loc[train_index]
            data_test.docdata=self.docdata.loc[test_index]

            dataList.append([data_train,data_test])

        return dataList

    def getInitialFoldIndex(self,fold_num=5):
        from sklearn import model_selection
        kfold = model_selection.KFold(n_splits=fold_num, shuffle=True)
        indexList = []
        Y=np.array(self.docdata["label"])
        Y = np.reshape(Y, newshape=(len(Y), 1))
        for train_index, test_index in kfold.split(Y):

            indexList.append([train_index,test_index])

        return indexList

class MyMetrics(Callback):
    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_cms=[]
        self.val_acc=[]

    def on_epoch_end(self, epoch, logs=None):

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
        if precision+recall==0:
            f1=0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        print("f1-score=", f1)

        acc = float(tp + tn) / float(tn + fp + fn + tp)

        print("accuracy=", acc)

        print()
        return

#model
class TwoInDNNModel:

    def __init__(self):
        self.name="TwoInputModel"
        self.model=None
        self.numEpoch=5

    def buildModel(self):
        '''
        define deep learning net
        :return:
        '''
        pass

    def saveModel(self):
        path=model_dir+self.name
        self.model.save(path)
    def trainModel(self,train,validate):
        import collections
        print(collections.Counter(train.dataY))
        watch_metrics = MyMetrics()

        print(self.name+" training")
        t0=time.time()

        cls_w={0:1,1:2}

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

    def predict(self,dataSet):
        em1, em2 = dataSet.dataEm1, dataSet.dataEm2
        feeddata = {"em1": em1, "em2": em2}

        Y=self.model.predict(feeddata,verbose=0)
        #print(Y)
        #Y=Y[0]
        Y=np.argmax(Y,axis=1)

        print(self.name,"finished predicting %d records"%len(em1))
        return Y


#model

class LSTMModel(TwoInDNNModel):

    def __init__(self):
        TwoInDNNModel.__init__(self)
        self.name="TwoInputLSTM"
        self.numEpoch=6

    def buildModel(self):
        datashape=(maxWords,features)

        #word net
        input1=layers.Input(shape=datashape,name="em1")
        input2=layers.Input(shape=datashape,name="em2")
        comLSTM=layers.LSTM(128)
        encode1=comLSTM(input1)
        encode2=comLSTM(input2)

        # extract net2

        L1_distance = lambda x: K.abs(x[0] - x[1])
        both = layers.merge([encode1, encode2], mode=L1_distance, output_shape=lambda x: x[0])

        hiddenLayer=layers.Dense(units=64,activation="relu")(both)
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
        #self.model.get_config()
        return self.model




#utilities
def transformDoc2Vec(corporus_docs):
    print("generate word embeddings (%d,%d)"%(maxWords,features))
    embeddings=[]

    for corporus_doc in corporus_docs:
        embedding=[]
        corporus_doc=corporus_doc.split(" ")
        n_count=min(maxWords,len(corporus_doc))
        for i in range(n_count):
            word=corporus_doc[i]
            if word in word_dict.keys():
                wordvec=word_dict[word]
            else:
                continue

            embedding.append(wordvec)
        n_count=len(embedding)
        for i in range(n_count,maxWords):
            embedding.append(np.zeros(shape=features))


        embeddings.append(embedding)

    embeddings=np.array(embeddings)

    return embeddings

def embedData(data):
    s1=np.array(data.docdata["sent1"])
    s2=np.array(data.docdata["sent2"])
    size=len(s1)
    sent=np.concatenate((s1,s2),axis=0)
    embeddings = transformDoc2Vec(sent)
    print("embeddded word shape",embeddings.shape)
    em1 = embeddings[:size]
    em2 = embeddings[size:]
    labels = np.array(data.docdata["label"], dtype=np.int)

    data.constructData(em1=em1, em2=em2, labels=labels)

    return data

def getFeedData(df):
    data = NLPDataSet()
    data.loadDocsData(df)
    data=embedData(data)

    return data

def loadWordDict(df):
    word_dict={}
    shape=df.shape
    features=shape[1]-1
    words_num=shape[0]
    wordvec=np.array(df.iloc[:,1:])
    words=np.array(df.iloc[:,0])
    print("words num%d, vecs shape"%words_num,shape)
    for i in range(words_num):

        word=words[i]
        vec=wordvec[i]
        word_dict[word]=vec

    return word_dict

#train model
def trainModel(MyModel,data1,data2=None):


    if data2 is not None and data2.size>10:
        print("data1 for train, data2 for validate")

        trainData = getFeedData(df=data1)

        validateData = getFeedData(df=data2)
        dnnmodel = MyModel()

        dnnmodel.trainModel(trainData, validateData)
        dnnmodel.saveModel()
        #return [
        #    [dnnmodel.getWeights(),dnnmodel.getStruct()]
        #]
    else:
        print("%d fold training"%model_num)

        trainData, validateData = NLPDataSet(), None
        trainData.loadDocsData(data1)


    dataList = trainData.getInitialFoldIndex(model_num)
    #modelWs=[]
    for i in range(model_num):
        # lstm dnn model
        dnnmodel = MyModel()

        dnnmodel.name += str(i)
        trainIndex, validateIndex = dataList[i]
        train=NLPDataSet()
        validate=NLPDataSet()
        train.docdata=trainData.docdata.iloc[trainIndex]
        validate.docdata=trainData.docdata.iloc[validateIndex]
        train=embedData(train)
        validate=embedData(validate)
        dnnmodel.trainModel(train, validate)
        dnnmodel.saveModel()
        print("\n==========%d/%d=================\n" % (i + 1, model_num))

        #modelWs.append([dnnmodel.getWeights(),dnnmodel.getStruct()])
    #return modelWs


word_dict=loadWordDict(df3)


trainModel(LSTMModel,df1,df2)

#output model info
'''
modelW_S=np.array(modelW_S)
print("model data",modelW_S.shape)
print(modelW_S)
modeldata=pd.DataFrame(data=modelW_S,columns=["modelweight","modelstruct"])
topai(1,modeldata)
'''


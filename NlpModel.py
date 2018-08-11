# -*- coding: UTF-8 -*-

from keras import models,layers,optimizers,losses,metrics
import time
from keras import utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import warnings
import numpy as np
import pandas as pd
from sklearn import metrics
from keras.callbacks import Callback
import keras.backend as K
import pickle,jieba,jieba.analyse
from sklearn import model_selection
warnings.filterwarnings("ignore")

class NLPDataSet:

    def __init__(self,testMode):

        self.docdata=None
        self.testMode=testMode

    def loadDicts(self):

        with open("./data/userdict-ch.txt","r") as f:
            for w in f:
                jieba.add_word(w.replace("\r","").replace("\n",""))

        self.stopwords=set()
        with open("./data/stopwords-ch.txt","r") as f:
            for w in f:
                self.stopwords.add(w.replace("\r","").replace("\n",""))

    def loadDocsData(self,ratio=1.0):

        inputfile="./data/train_nlp_data.csv"

        with open(inputfile,"r") as f:
            records=[]
            for line in f:
                record=line.replace("\n","").replace("\r", "").split("\t")
                records.append(record)
        if self.testMode:
            self.docdata=pd.DataFrame(data=records,columns=["id","sent1","sent2"])
        else:
            self.docdata=pd.DataFrame(data=records,columns=["id","sent1","sent2","label"])

        n=len(self.docdata)
        n=int(ratio*n)
        self.docdata=self.docdata.iloc[:n]
        self.loadDicts()
        cleanVec=np.vectorize(self.cleanDoc)
        self.docdata["sent1"]=cleanVec(self.docdata["sent1"])
        self.docdata["sent2"]=cleanVec(self.docdata["sent2"])

        print("loaded %d records"%len(self.docdata["id"]))

    def cleanDoc(self,doc):

        cut_words=jieba.cut_for_search(doc)
        words=[]
        for word in cut_words:
            if word in self.stopwords:
                continue
            words.append(word)
        words=" ".join(words)

        return words

    def constructData(self):
        """

        :param s1:
        :param s2:
        :param labels:
        :return:
        """
        s1 = np.array(self.docdata["sent1"])
        s2 = np.array(self.docdata["sent2"])
        self.text1=s1
        self.text2=s2
        if self.testMode:
            self.label=np.zeros(shape=len(s1),dtype=np.int)
        else:
            self.label = np.array(self.docdata["label"],dtype=np.int)

    def getFold(self,fold_num=5,index=0):

        with open(model_dir+"indexFold"+str(fold_num),"rb") as f:
            kfold=pickle.load(f)

        train_index, test_index =kfold[index]

        data_train=NLPDataSet(False)
        data_test=NLPDataSet(False)

        data_train.text1,data_train.text2,data_train.label=\
                self.text1[train_index],self.text2[train_index],self.label[train_index]

        data_test.text1,data_test.text2,data_test.label=\
                self.text1[test_index],self.text2[test_index],self.label[test_index]


        return data_train,data_test


    def genFoldIndex(self,fold_num=5):

        kfold = model_selection.KFold(n_splits=fold_num, shuffle=True)
        dataList = []
        Y=np.array(self.docdata["label"])
        Y = np.reshape(Y, newshape=(len(Y), 1))
        for train_index, test_index in kfold.split(Y):
            dataList.append([train_index,test_index])

        with open(model_dir+"indexFold"+str(fold_num),"wb") as f:
            pickle.dump(dataList,f)

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
        if precision*recall==0:
            f1=0
        else:
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

def selectTopK(text,topK=100):
    #relativewords=text.split(" ")
    cutwords=text.split(" ")
    if len(cutwords)<=topK:
        return text

    try:
        relativewords=jieba.analyse.textrank(sentence=text,topK=topK)

    except:
        print("textract top k keywords failed")
        print(text)

        return text

    words=[]
    lowIndex=[]
    n=0
    for w in cutwords:
        if w not in relativewords:
            lowIndex.append(n)

        words.append(w)

        if len(words)>topK:
            rm=lowIndex[0]
            del words[rm]
            del lowIndex[0]

        n+=1

    return " ".join(words)

class NlpModel:
    tokenModel=Tokenizer(num_words=10000)
    Size_Vocab=10000

    def __init__(self):
        self.name="NlpLSTMDecisionModel"
        self.model=None
        self.numEpoch=1
        self.maxLen=64
        self.embeddingSize=200
        self.class_weight={0:1.0,1:1.7}
        self.batch_size=64


        self.model_dir=model_dir

    def buildModel(self):

        #word net
        input1=layers.Input(shape=(self.maxLen,),name="seq1")
        input2=layers.Input(shape=(self.maxLen,),name="seq2")

        comEmbedding=layers.Embedding(input_dim=self.Size_Vocab,output_dim=self.embeddingSize,input_length=self.maxLen)
        emb1=comEmbedding(input1)
        emb2=comEmbedding(input2)

        comLSTM=layers.Bidirectional(layers.LSTM(256,return_sequences=True))
        encode1=comLSTM(emb1)
        encode2=comLSTM(emb2)

        comFlat=layers.Flatten()
        encode1=comFlat(encode1)
        encode2=comFlat(encode2)
        # extract net2

        L1_distance = lambda x: K.abs(x[0] - x[1])
        both = layers.merge([encode1, encode2], mode=L1_distance, output_shape=lambda x: x[0])

        hiddenLayer=layers.Dense(units=150,activation="relu")(both)
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

    def text2Seq(self,texts):

        relative=np.vectorize(selectTopK)

        try:
            #texts=relative(texts,self.maxLen)
            pass
        except:
            print("top K extraction failed")
            print(texts)

        seq=self.tokenModel.texts_to_sequences(texts)

        seq=sequence.pad_sequences(seq,self.maxLen)

        print("generated seqs")

        return seq

    def trainModel(self,train,validate):
        from keras.callbacks import TensorBoard
        import collections
        print(collections.Counter(train.label))
        #tensorboard = TensorBoard(log_dir="/tmp/zhangzyTFK",histogram_freq=1)
        watch_metrics = MyMetrics()
        print(self.name+" training")
        print("class weight",self.class_weight)

        #siamese networks
        t0=time.time()

        label = train.label

        text1,text2=train.text1,train.text2
        seq1=self.text2Seq(text1)
        seq2=self.text2Seq(text2)

        feeddata={"seq1":seq1,"seq2":seq2}
        feedlabel={
            "label":utils.to_categorical(label,2)
        }
        print("transformed train data to seqs")

        if validate is not None:

            vtext1=validate.text1
            vtext2=validate.text2
            vseq1=self.text2Seq(vtext1)
            vseq2=self.text2Seq(vtext2)
            val_label = validate.label

            val_feeddata = {"seq1":vseq1,"seq2":vseq2}
            val_feedlabel = {
                "label": utils.to_categorical(val_label, 2)
            }
            val_data=(val_feeddata,val_feedlabel)
            print("transformed validate data to seqs")
            
        else:

            val_data=None


        self.model.fit(feeddata,feedlabel,
            verbose=2, epochs=self.numEpoch, batch_size=self.batch_size

                       ,validation_data=val_data
                       ,class_weight={"label":self.class_weight}
                       ,callbacks=[watch_metrics]
                       #,validation_split=0.2
                       )

        t1=time.time()

        print("siamese model(%s) training finished in %ds"%(self.name,t1-t0))

        #print(self.model.get_weights())

    def predict(self,dataSet):
        text1,text2=dataSet.text1, dataSet.text2
        seq1=self.text2Seq(text1)
        seq2=self.text2Seq(text2)
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


def trainModel():


    data.constructData()

    buildTokenizer(NlpModel.tokenModel,data)


    dnnmodel.name += str(model_index)

    train, test = data.getFold(model_num,model_index)

    try:
        dnnmodel.loadModel()
        print("model hot start")
    except:
        dnnmodel.buildModel()
        print("model hot start failed")

    dnnmodel.trainModel(train, test)


    dnnmodel.saveModel()

    print("\n==========%d/%d(model training finished)=================\n" % (model_index + 1, model_num))

if __name__ == '__main__':
    model_dir="./models/"
    model_num = 10
    model_index=0

    data=NLPDataSet(False)

    data.loadDocsData(1)
    #data.docdata=df1

    #data.genFoldIndex(model_num);


    dnnmodel=NlpModel()
    trainModel()


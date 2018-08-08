import pandas as pd
from keras import models,optimizers,losses
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import pickle
import jieba.analyse
#
class NLPDataSet:

    def __init__(self,testMode):

        self.docdata=None
        self.testMode=testMode



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

def buildTokenizer(tokenModel,data):
        #tkonizer model
        text1,text2=data.text1,data.text2
        texts=np.concatenate((text1,text2),axis=0)
        tokenModel.fit_on_texts(texts)

def selectTopK(corporu,topK=100):
    relativewords=jieba.analyse.textrank(sentence=corporu,topK=topK)
    cutwords=corporu.split(" ")
    words=[]
    for w in cutwords:
        if w not in relativewords:
            continue
        words.append(w)

    return " ".join(words)

class NlpModel:
    tokenModel=Tokenizer(num_words=10000)
    Size_Vocab=10000

    def __init__(self):
        self.name="NlpLSTMDecisionModel"
        self.model=None
        self.numEpoch=1
        self.maxLen=96
        self.embeddingSize=196
        self.class_weight={0:1,1:4.5}
        self.batch_size=64

        self.model_dir=model_dir


    def predict(self,dataSet):
        text1,text2=dataSet.text1, dataSet.text2
        text1=relative(text1,self.maxLen)
        text2=relative(text2,self.maxLen)
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




def getTestData(df):
    data = NLPDataSet(True)
    data.docdata=df
    data.constructData()

    return data

def getModels():
    classifiers=[]
    for i in range(modelNum):

        classifier=NlpModel()
        classifier.name=modelName+str(i)
        classifier.loadModel()
        classifiers.append(classifier)

    print("loaded %d Models"%modelNum)
    return classifiers

def ensemBleTest(df):
    print("\n ============begin to test=========== \n")

    IDs=np.array(df["id"],dtype=np.int)

    resultsList=[]
    for predictor in classifiers:

        predicts = predictor.predict(data)
        resultsList.append(predicts)

    results=np.sum(resultsList,axis=0)
    vote_c=modelNum//2
    results = np.array(results>vote_c, dtype=np.int)

    results=pd.DataFrame(data={"id":IDs,"label":results})
    return results


#main run
if __name__ == '__main__':
    model_dir="./models/"
    modelName="NlpLSTMDecisionModel"
    modelNum=10
    relative=np.vectorize(selectTopK)
    data=getTestData(df1)
    classifiers=getModels()

    results=ensemBleTest(df1)

    topai(1,results)

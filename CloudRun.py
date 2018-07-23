import pandas as pd
from keras import models,optimizers,losses
import numpy as np
import json

maxWords=30
features=128
modelNum=10
#
class NLPDataSet:
    def __init__(self):
        pass
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
#model
class TwoInDNNModel:
    def __init__(self):
        self.name="TwoInputModel"

    def buildModel(self):
        '''
        define deep learning net
        :return:
        '''
        self.model.compile(optimizer=optimizers.Adam(),
                      loss={
                          "label":losses.binary_crossentropy
                           }
                      )
        return self.model

    def loadModelStructure(self,model_struct):
        #print("build model from", model_struct)
        self.model=models.model_from_json(model_struct)


    def setWeight(self,str_w):

        ws=json.loads(str_w)
        for i in range(len(ws)):
            ws[i]=np.array(ws[i])
        self.model.set_weights(ws)

    def predict(self,dataSet):
        em1, em2 = dataSet.dataEm1, dataSet.dataEm2
        feeddata = {"em1": em1, "em2": em2}

        Y=self.model.predict(feeddata,verbose=0)
        #print(Y)
        #Y=Y[0]
        Y=np.argmax(Y,axis=1)

        print(self.name,"finished predicting %d records"%len(em1))
        return Y


#utility
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

def getTestData(df):
    data = NLPDataSet()
    s1=np.array(df["sent1"])
    s2=np.array(df["sent2"])
    size=len(s1)
    sent=np.concatenate((s1,s2),axis=0)
    embeddings = transformDoc2Vec(sent)

    em1 = embeddings[:size]
    em2 = embeddings[size:]
    labels = np.zeros(shape=size, dtype=np.int)

    data.constructData(em1=em1, em2=em2, labels=labels)

    return data

def getModels(df):
    classifiers=[]
    modelWs=np.array(df["modelweight"])
    modelStruct=np.array(df["modelstruct"])
    modelNum=len(modelWs)
    for i in range(modelNum):
        mw=modelWs[i]
        ms=modelStruct[i]

        classifier=TwoInDNNModel()
        classifier.loadModelStructure(ms)
        classifier.buildModel()
        classifier.setWeight(mw)
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

#main run


word_dict=loadWordDict(df2)
data=getTestData(df1)
classifiers=getModels(df3)
modelNum=len(classifiers)
results=ensemBleTest(df1)

topai(1,results)

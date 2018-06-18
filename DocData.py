# coding=utf-8
import codecs

import jieba, jieba.analyse
import pandas as pd
import json
import numpy as np
import gensim
from gensim.models.doc2vec import LabeledSentence

class DocDatapreprocessing:
    def __init__(self,inputfile,outputfile):
        self.inputPath=inputfile
        self.outputPath=outputfile
        self.features=10
        self.dics=None
        self.docModel=None
        print "created doc data loader"

    def loadDocsData(self):

        with open(self.inputPath,"r") as f:
            records=[]
            for line in f:
                record=line.replace("\r\n", "").split("\t")
                #record[3]=int(record[3])
                records.append(record)

        self.docdata=pd.DataFrame(data=records,columns=["no","s1","s2","label"])
        print "data\n",self.docdata.head(3)

        with open("./data/words.json","r") as f:
            words=json.load(f)
        #print(words)
        self.excludes=words["excludes"]
        self.stopwords=[unicode(x,"utf-8") for x in words["stopwords"]]

        newwords=words["newwords"]
        for w in newwords:
            jieba.add_word(w)

        #print self.excludes,"\n",self.stopwords,"\n",newwords

    def cleanDocs(self,docs=None):

        if docs is None:
            s1, s2 = self.docdata["s1"], self.docdata["s2"]
            docs = s1.append(s2)


        docX=[]

        for doc in docs:
            cut_words=jieba.cut_for_search(doc)
            words=[]
            for w in cut_words:

                if w in self.excludes or w in self.stopwords:
                    continue
                words.append(w)

            docX.append(" ".join(words))

        with codecs.open(filename="./data/sentences.txt",mode="w",encoding="utf-8") as f:
            for x in docX:
                f.write(x+"\n")

        return docX

    def trainDocModel(self, epoch_num=1):
        # 实例DM和DBOW模型
        self.cleanDocs()
        corporus=gensim.models.doc2vec.TaggedLineDocument("./data/sentences.txt")

        model_dm = gensim.models.Doc2Vec(min_count=1, window=10, vector_size=self.features,workers=4 )
        model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, vector_size=self.features,workers=4)

        # 使用所有的数据建立词典
        model_dm.build_vocab(corporus)
        model_dbow.build_vocab(corporus)

        # 进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
        for epoch in range(epoch_num):

            model_dm.train(corporus,model_dm.corpus_count,epochs=model_dm.epochs)
            model_dbow.train(corporus,model_dbow.corpus_count,epochs=model_dbow.epochs)

        self.docModel=[model_dm,model_dbow]

        model_dm.save("./data/model_dm")
        model_dbow.save("./data/model_dbow")

        print("doc2vec model training finished")

        return model_dm, model_dbow


    def loadModel(self):
        self.docModel=[
            gensim.models.Doc2Vec.load("./data/model_dm"),
            gensim.models.Doc2Vec.load("./data/model_dbow")
        ]

    def transformDoc2Vec(self):
        self.cleanDocs()
        coporus=gensim.models.doc2vec.TaggedLineDocument("./data/sentences.txt")
        model_dm,model_dbow=self.docModel
        px1=[]
        px2=[]
        for doc in coporus:
            #print(doc)
            if len(doc[0])==0:
                px1.append(np.zeros(self.features))
                px2.append(np.zeros(self.features))
                continue
            px1.append(model_dm[doc])
            px2.append(model_dbow[doc])
        print(len(px1),len(px2),len(px1[0]),len(px2[0]))
        return np.array(px1),np.array(px2)

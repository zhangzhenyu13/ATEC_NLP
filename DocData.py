# coding=utf-8
import codecs

import jieba, jieba.analyse
import pandas as pd
import numpy as np
import gensim
import time
import initConfig

class DocDatapreprocessing:
    def __init__(self,inputfile):
        self.inputPath=inputfile
        self.features=initConfig.config["features"]
        self.model_dm=None
        self.model_dbow=None
        print "created doc data loader"

    def loadDocsData(self):

        with open(self.inputPath,"r") as f:
            records=[]
            for line in f:
                record=line.replace("\n","").replace("\r", "").split("\t")
                records.append(record)

        self.docdata=pd.DataFrame(data=records,columns=["no","s1","s2","label"])
        #print "data\n",self.docdata.head(3)

        #print(words)
        self.excludes=initConfig.config["excludes"]
        self.stopwords=[unicode(x,"utf-8") for x in initConfig.config["stopwords"]]

        newwords=initConfig.config["newwords"]
        for w in newwords:
            jieba.add_word(w)

        #print self.excludes,"\n",self.stopwords,"\n",newwords

    def cleanDocs(self,docs=None):

        if docs is None:
            s1, s2 = self.docdata["s1"], self.docdata["s2"]
            docs = s1.append(s2)

        corpo_docs=[]

        for doc in docs:
            cut_words=jieba.cut_for_search(doc)
            words=[]
            for w in cut_words:

                if w in self.excludes or w in self.stopwords:
                    continue
                words.append(w)

            corpo_docs.append(words)

        return corpo_docs

    def trainDocModel(self, epoch_num=3):
        # 实例DM和DBOW模型
        t0=time.time()
        corpo_docs=self.cleanDocs()

        corporus=[]
        for i in range(len(corpo_docs)):
            corpo_doc=corpo_docs[i]
            tagdoc=gensim.models.doc2vec.TaggedDocument(words=corpo_doc,tags=[i])
            corporus.append(tagdoc)

        self.model_dm = gensim.models.Doc2Vec(min_count=1, window=10, vector_size=self.features,workers=4 ,dm=1)
        self.model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, vector_size=self.features,workers=4,dm=0)

        # 使用所有的数据建立词典
        self.model_dm.build_vocab(corporus)
        self.model_dbow.build_vocab(corporus)

        # 进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度


        self.model_dm.train(corporus,total_examples=len(corpo_docs),epochs=epoch_num)
        self.model_dbow.train(corporus,total_examples=len(corpo_docs),epochs=epoch_num)
        t1=time.time()
        print("doc2vec model training finished in %d s"%(t1-t0))

    def saveModel(self):
        self.model_dm.save("./models/model_dm-"+str(self.features))
        self.model_dbow.save("./models/model_dbow-"+str(self.features))
        print("saved doc2vec model")

    def loadModel(self):

        self.model_dm=gensim.models.Doc2Vec.load("./models/model_dm-"+str(self.features))
        self.model_dbow=gensim.models.Doc2Vec.load("./models/model_dbow-"+str(self.features))

        print("loaed doc2vec model")
        #print(type(self.model_dbow),type(self.model_dm))

    def transformDoc2Vec(self,docs):
        print("generate doc vecs")
        px1=[]
        px2=[]

        corporus_docs=self.cleanDocs(docs)

        for corporus_doc in corporus_docs:

            doc1=self.model_dbow.infer_vector(corporus_doc)
            doc2=self.model_dm.infer_vector(corporus_doc)
            #print(i,doc1,doc2)
            #count+=1
            #if count>10:
            #    break

            px1.append(doc1)
            px2.append(doc2)
            #print(model_dbow[doc])
        #print(px1[0])
        #print(model_dbow[u"不 记得 花呗 账号 怎么 怎么办"])
        px1=np.array(px1)
        px2=np.array(px2)
        px=np.concatenate((px1,px2),axis=1)
        #print(len(px1),len(px2),len(px1[0]),len(px2[0]))
        #print(px1[:2])
        #print()
        #print(px2[:2])
        return px


# coding=utf-8

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
        newwords = initConfig.config["newwords"]
        for w in newwords:
            jieba.add_word(w)
        print "init doc model"


    def cleanDocs(self,docs):

        corpo_docs=[]

        for doc in docs:
            cut_words=jieba.cut_for_search(doc)
            words=[]
            for w in cut_words:

                words.append(w)

            corpo_docs.append(words)

        return corpo_docs

    def trainDocModel(self, docs,epoch_num=500):
        t0=time.time()
        corpo_docs=self.cleanDocs(docs)

        corporus=[]
        for i in range(len(corpo_docs)):
            corpo_doc=corpo_docs[i]
            tagdoc=gensim.models.doc2vec.TaggedDocument(words=corpo_doc,tags=[i])
            corporus.append(tagdoc)

        self.model_dm = gensim.models.Doc2Vec(min_count=1, window=5, vector_size=self.features,workers=12 ,dm=1)

        self.model_dm.build_vocab(corporus)

        self.model_dm.train(corporus,total_examples=len(corpo_docs),epochs=epoch_num)
        t1=time.time()
        print("doc2vec model training finished in %d s"%(t1-t0))

    def transformDoc2Vec(self,docs):
        print("generate doc vecs")
        px=[]

        corporus_docs=self.cleanDocs(docs)

        for corporus_doc in corporus_docs:

            doc=self.model_dm.infer_vector(corporus_doc)

            px.append(doc)

        px=np.array(px)

        return px

    def saveModel(self):
        self.model_dm.save("./models/model_dm")
        print("saved doc2vec model")

    def loadModel(self):
        self.model_dm=gensim.models.Doc2Vec.load("./models/model_dm")
        print("loaed doc2vec model")



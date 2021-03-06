# coding=utf-8

import jieba, jieba.analyse
import numpy as np
import gensim
import time
import initConfig

class WordEmbedding:
    def __init__(self):

        self.features=initConfig.config["features"]
        self.model=None
        self.maxWords=initConfig.config["maxWords"]
        self.buildVoca=True
        self.stopwords=[]
        with open("userdict-ch.txt","r") as f:
            for w in f:
                jieba.add_word(w.replace("\r","").replace("\n",""))
        with open("stopwords-ch.txt","r") as f:
            for w in f:
                self.stopwords.append(w.replace("\r","").replace("\n",""))

        self.model = gensim.models.Word2Vec(size=self.features, window=10,min_count=2)

        print "init word model"


    def cleanDocs(self,docs):

        corpo_docs=[]

        for doc in docs:
            cut_words=jieba.cut_for_search(doc)
            words=[]
            for w in cut_words:
                if w==u" " or w==u"":continue
                words.append(w)

            corpo_docs.append(words)

        return corpo_docs

    def trainDocModel(self, docs,epoch_num=50):
        t0=time.time()

        corpo_docs = self.cleanDocs(docs)

        if self.buildVoca:
            self.model.build_vocab(corpo_docs)


        self.model.train(corpo_docs,total_examples=len(docs),epochs=epoch_num)

        t1=time.time()
        print("word2vec model training finished in %d s"%(t1-t0))

    def transformDoc2Vec(self,docs):
        print("generate word embeddings")
        embeddings=[]

        corporus_docs=self.cleanDocs(docs)

        for corporus_doc in corporus_docs:
            embedding=np.zeros(shape=(self.maxWords,self.features))
            n_count=min(self.maxWords,len(corporus_doc))
            for i in range(n_count):
                word=corporus_doc[i]
                if word in self.stopwords:
                    continue

                try:
                    wordvec=self.model[word]
                except:
                    continue

                embedding[i]=wordvec


            embeddings.append(embedding)

        embeddings=np.array(embeddings)

        return embeddings


    def saveModel(self):
        self.model.save("./models/word2vec")
        print("saved word2vec model")

    def loadModel(self):
        self.buildVoca=False
        self.model=gensim.models.Doc2Vec.load("./models/word2vec")
        print("loaded word2vec model")

if __name__ == '__main__':
    from FeatureDataSet import NLPDataSet

    data = NLPDataSet(testMode=False)
    data.loadDocsData("../data/train_nlp_data.csv")


    docs = data.getAllDocs()
    docs=list(docs)

    '''
    import scipy.stats as sts
    maxW=0
    minW=20
    wl=set()
    for ws in docs:
        if len(ws)>maxW:
            maxW=len(ws)
        if len(ws)<minW:
            minW=len(ws)
        wl.add(len(ws))
    print(maxW,minW)
    wl=list(wl)
    print(np.mean(wl),np.var(wl),np.median(wl),sts.mode(wl))
    wl.sort()
    import matplotlib.pyplot as plt
    x=np.arange(len(wl))
    y=np.array(wl)
    plt.plot(x,y)
    plt.show()
    exit(10)
    '''

    #docModel = WordEmbedding();docModel.cleanDocs(docs);exit(10)
    '''
    docs_add=[]
    with open("../data/wiki-chs.txt","r") as f:
        docs2=f.read().split("\n")

        for doc in docs2:
            docs_add.append(doc)
    import random
    random.shuffle(docs_add)
    docs_add=docs_add[:int(0.3*len(docs_add))]

    print("wiki data records=%d"%len(docs_add))

    #wiki corporus
    '''
    wordModel = WordEmbedding()
    wordModel.trainDocModel(docs,100)
    print(wordModel.model.wv.vocab)
    exit(10)
    wordModel.saveModel()
    wordModel.loadModel()
    vecs=wordModel.transformDoc2Vec(docs)
    print(vecs.shape)



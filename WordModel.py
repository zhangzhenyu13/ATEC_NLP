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
        self.padding=0#0 right 1 left
        self.maxWords=initConfig.config["maxWords"]
        newwords = initConfig.config["newwords"]
        for w in newwords:
            jieba.add_word(w)

        self.model = gensim.models.Word2Vec(size=self.features, min_count=1)

        print "init word model"


    def cleanDocs(self,docs):

        corpo_docs=[]

        for doc in docs:
            cut_words=jieba.cut_for_search(doc)
            words=[]
            for w in cut_words:

                words.append(w)

            corpo_docs.append(words)

        return corpo_docs

    def trainDocModel(self, docs,docs_add=None,epoch_num=50):
        t0=time.time()


        allDocs=docs
        if docs_add is not None:
            allDocs=docs+docs_add
        sumN=len(allDocs)
        speN=len(docs)

        corpo_docs = self.cleanDocs(allDocs)

        self.model.build_vocab(corpo_docs)
        if speN<sumN:
            self.model.train(corpo_docs[len(docs):],total_examples=sumN-speN,epochs=1)

        self.model.train(corpo_docs[:len(docs)],total_examples=speN,epochs=epoch_num)

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
                try:
                    wordvec=self.model[word]
                except:
                    continue
                if self.padding==0:
                    embedding[i]=wordvec
                else:
                    embedding[-i]=wordvec

            embeddings.append(embedding)

        embeddings=np.array(embeddings)

        return embeddings

    def saveModel(self):
        self.model.save("./models/word2vec")
        print("saved word2vec model")

    def loadModel(self):
        self.model=gensim.models.Doc2Vec.load("./models/word2vec")
        print("loaed word2vec model")

if __name__ == '__main__':
    from FeatureDataSet import NLPDataSet

    data = NLPDataSet(testMode=False)
    data.loadDocsData("../data/train_nlp_data.csv")
    with open("../data/wiki-chs.txt","r") as f:
        docs2=f.read().split("\n")

    docs = data.getAllDocs()
    docs=list(docs)

    docModel = WordEmbedding()

    docModel.trainDocModel(docs,docs2)
    docModel.saveModel()

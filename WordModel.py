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

    def trainDocModel(self, docs,epoch_num=20):
        t0=time.time()
        corpo_docs=self.cleanDocs(docs)

        self.model = gensim.models.Word2Vec(size=self.features,min_count=1)

        self.model.build_vocab(corpo_docs)

        self.model.train(corpo_docs,total_examples=len(corpo_docs),epochs=epoch_num)
        t1=time.time()
        print("doc2vec model training finished in %d s"%(t1-t0))

    def transformDoc2Vec(self,docs):
        print("generate doc vecs")
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

<<<<<<< HEAD
if __name__ == '__main__':
    from FeatureDataSet import WordDataSet

    data = WordDataSet(testMode=False)
    data.loadDocsData("../data/train_nlp_data.csv")
    docs = data.getAllDocs()

    docModel = WordEmbedding()
    docModel.loadModel()
=======

>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13

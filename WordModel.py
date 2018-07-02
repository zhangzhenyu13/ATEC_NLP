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
        newwords = initConfig.config["newwords"]
        for w in newwords:
            jieba.add_word(w)

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
    docModel = WordEmbedding()
    docModel.trainDocModel(docs,100)
    docModel.saveModel()



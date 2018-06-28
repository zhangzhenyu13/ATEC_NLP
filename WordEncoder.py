import jieba, jieba.analyse
import numpy as np
import time
import initConfig


class WordEncoder:
    def __init__(self):

        self.model = None
        self.buildVoca = True
        newwords = initConfig.config["newwords"]
        for w in newwords:
            jieba.add_word(w)

        self.vocab={}
        print "init word encoder"

    def cleanDocs(self, docs):

        corpo_docs = []

        for doc in docs:
            cut_words = jieba.cut_for_search(doc)
            words = []
            for w in cut_words:
                if w == u" " or w == u"": continue
                words.append(w)

            corpo_docs.append(words)

        return corpo_docs

    def trainDocModel(self, docs):
        t0=time.time()

        vocab= jieba.analyse.textrank(". ".join(docs),topK=10000)
        for i in range(6500):
            self.vocab[vocab[i]]=i+1

        print("build vocab size=%d"%len(self.vocab))
        t1=time.time()
        print("word2vec model training finished in %d s"%(t1-t0))

    def transformDoc2Vec(self,docs):
        print("generate word encoding")
        seqs=np.zeros(shape=(len(docs),10),dtype=np.int)

        corporus_docs=self.cleanDocs(docs)
        i=0
        for corporus_doc in corporus_docs:
            pos=0
            for word in corporus_doc:
                if word in self.vocab.keys():
                    seqs[i][pos]=self.vocab[word]
                pos+=1
                if pos>=10:
                    break
            i+=1

        return seqs

    def saveModel(self):
        import pickle
        with open("./models/wordencoder","wb") as f:

            pickle.dump(self.vocab,f)
        print("saved wordencoder model")

    def loadModel(self):
        import pickle
        with open("./models/wordencoder","rb") as f:
            self.vocab=pickle.load(f)
        print("loaded wordencoder model")

if __name__ == '__main__':
    from FeatureDataSet import NLPDataSet

    data = NLPDataSet(testMode=False)
    data.loadDocsData("../data/train_nlp_data.csv")

    docs = data.getAllDocs()
    docs = list(docs)

    w_encoder=WordEncoder()
    w_encoder.trainDocModel(docs)
    w_encoder.saveModel()
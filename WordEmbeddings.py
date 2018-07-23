import pandas as pd
import numpy as np
import gensim
import time
features=256
minCount=2
def getCorporaus(docs):
    corpo_docs=[]
    for s in docs:
        corpo_docs.append(s.split(" "))

    return corpo_docs

def trainDocModel(docs,epoch_num=50):
    t0=time.time()
    corpo_docs=getCorporaus(docs)
    wordModel = gensim.models.Word2Vec(size=features, window=10,min_count=minCount)

    wordModel.build_vocab(corpo_docs)

    wordModel.train(corpo_docs,total_examples=len(corpo_docs),epochs=epoch_num)

    t1=time.time()
    print("word2vec model training finished in %d s"%(t1-t0))

    return wordModel



#df1 contain splited data
s1=np.array(df1["sent1"])
s2=np.array(df1["sent2"])
sent=np.concatenate((s1,s2),axis=0)
wordModel=trainDocModel(sent,100)
word2vecmodelpath=model_dir+"word2vec-"+str(features)
wordModel.save(word2vecmodelpath)
words=wordModel.wv.vocab.keys()
word_dict=[]
for word in words:
    vec=list(wordModel[word])
    vec.insert(0,word)
    word_dict.append(vec)


cols=["f%d"%d for d in range(features)]
cols.insert(0,"words")

word_dict=pd.DataFrame(data=word_dict,columns=cols)
worddictpath=model_dir+"word2vecDict-"+str(features)+".pkl"
word_dict.to_pickle(worddictpath)

topai(1,word_dict)


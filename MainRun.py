# coding=utf-8
import argparse
from WordModel import WordEmbedding
from DocModel import DocDatapreprocessing
from FeatureDataSet import NLPDataSet
import numpy as np
from LSTMDNN import TwoInDNNModel


#test phase
def testPhase():
    print("\n ============begin to test=========== \n")


    data = NLPDataSet(testMode=True)
    data.loadDocsData(inputfile)
    docs = data.getAllDocs()

    # embedding words
    emModel = WordEmbedding()
    emModel.loadModel()

    embeddings = emModel.transformDoc2Vec(docs)

    # embedding docs
    docModel = DocDatapreprocessing()
    docModel.loadModel()

    sentences = docModel.transformDoc2Vec(docs)

    n_count = len(embeddings)
    em1 = embeddings[:n_count // 2]
    em2 = embeddings[n_count // 2:]
    s1 = sentences[:n_count // 2]
    s2 = sentences[n_count // 2:]

    labels = np.zeros(shape=n_count,dtype=np.int)

    data.constructData(s1=s1, s2=s2, em1=em1, em2=em2, labels=labels)

    classifier = TwoInDNNModel()

    classifier.loadModel()

    results=classifier.predict(data)

    no=data.docdata["no"]

    with open(outputfile,"w") as f:
        for i in range(len(results)):
            results[i]=1-results[i]

            f.write(str(no[i])+"\t"+str(results[i])+"\n")

if __name__ == '__main__':
    # parse input and output file path
    inputfile = "../data/validate.csv"
    outputfile = "../data/results.csv"

    parser = argparse.ArgumentParser(description="feed input and output files")
    parser.add_argument('--input', dest="input", default=inputfile, help='path of input file')
    parser.add_argument('--output', dest="output", default=outputfile, help='path of output file')

    args = parser.parse_args()
    inputfile = args.input
    outputfile = args.output

    #run test
    testPhase()

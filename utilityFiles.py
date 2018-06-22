import pandas as pd
from sklearn import metrics
import numpy as np
import collections

def splitTrainValidate(inputfile, ratio):
    lines = []
    with open(inputfile, "r") as f:
        for line in f:
            lines.append(line)

    import random
    n_count = len(lines)
    print("shuffle %d records" % n_count)
    random.shuffle(lines)
    train = lines[:int(ratio * n_count)]
    validate = lines[int(ratio * n_count):]

    #train data
    with open("./data/train.csv", "w") as f:
        for line in train:
            f.write(line)

    #validate data
    labels=[]
    with open("./data/validate.csv", "w") as f:
        for line in validate:
            record=line.split("\t")
            labels.append(record[3])
            f.write("\t".join(record[:3])+"\n")

    with open("./data/validatelabels.csv","w") as f:
        for r in labels:
            f.write(r)


def testScore(labelfile,resultfile):

    with open(resultfile, "r") as f:
        records = []
        for line in f:
            record = line.replace("\n", "").replace("\r", "").split("\t")
            records.append(record)

    results = pd.DataFrame(data=records, columns=["no", "predictions"])

    predicts=np.array(results["predictions"],dtype=np.int)

    labels=[]
    with open(labelfile, "r") as f:
        for r in f:
            labels.append(
                int(r.replace("\n","").replace("\r",""))
            )
    labels=np.array(labels,dtype=np.int)

    print("labels",collections.Counter(labels))
    print("predicts",collections.Counter(predicts))
    #measure
    f1=metrics.f1_score(labels,predicts)

    print("f1-score=",f1)

    acc=metrics.accuracy_score(labels,predicts)

    print("accuracy=",acc)

if __name__ == '__main__':
    #splitTrainValidate("./data/train_nlp_data.csv",0.8)
    testScore("./data/validatelabels.csv","./data/results.csv")
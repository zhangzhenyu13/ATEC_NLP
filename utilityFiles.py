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
    with open("../data/train.csv", "w") as f:
        for line in train:
            f.write(line)

    #validate data
    labels=[]
    with open("../data/validateX.csv", "w") as f:
        for line in validate:
            record=line.split("\t")
            labels.append([record[0],record[3]])
            f.write("\t".join(record[:3])+"\n")

    with open("../data/validateY.csv","w") as f:
        for r in labels:
            f.write("\t".join(r))

    with open("../data/validate.csv","w") as f:
        for line in validate:
            f.write(line)

    print("finished spliting")

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
                r.replace("\n","").replace("\r","").split("\t")
            )

    labels=pd.DataFrame(data=labels,columns=["no","labels"])
    labels=np.array(labels["labels"],dtype=np.int)

    print("labels",collections.Counter(labels))
    print("predicts",collections.Counter(predicts))

    CM=metrics.confusion_matrix(y_true=labels,y_pred=predicts,labels=[0,1])
    tn,fp,fn,tp=CM.ravel()

    precision=float(tp)/float(tp+fp)
    recall=float(tp)/float(fn+tp)
    print("tp,fp,tn,fn",tp,fp,tn,fn)
    print(CM)
    print("precision,recall",precision,recall)
    #measure
    f1=2*precision*recall/(precision+recall)

    print("f1-score=",f1)

    acc=float(tp+tn)/float(tn+fp+fn+tp)

    print("accuracy=",acc)

if __name__ == '__main__':
    testScore("../data/validateY.csv","../data/results.csv")


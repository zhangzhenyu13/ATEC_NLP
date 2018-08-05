from sklearn import metrics
import numpy as np
import collections
import pandas as pd

def testScore(labels,predicts):
    labels.sort_index(by=['id'])
    predicts.sort_index(by=['id'])

    predicts=np.array(predicts["label"],dtype=np.int)
    labels=np.array(labels["label"],dtype=np.int)

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
    testScore(df1,df2)


from keras import models,layers,optimizers,losses
import time,keras.backend as K
from keras import utils
import warnings
import numpy as np
import collections
import initConfig

warnings.filterwarnings("ignore")

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#model
class TwoInDNNModel:

    def __init__(self):
        self.name="TwoInputDNN"

    def buildModel(self):
        datashape=(initConfig.config["maxWords"],initConfig.config["features"])

        outputDim=2
        #word net
        input1=layers.Input(shape=datashape,name="em1")
        input2=layers.Input(shape=datashape,name="em2")
        comLSTM=layers.LSTM(64)
        encode1=comLSTM(input1)
        encode2=comLSTM(input2)
        features=layers.concatenate([encode1,encode2],axis=-1)

        # extract net2

        hiddenLayer=layers.Dense(units=128,activation="relu")(features)
        predictionLayer=layers.Dense(units=outputDim,activation="relu",name="label")(hiddenLayer)

        self.model=models.Model(inputs=[input1,input2],outputs=[predictionLayer])

        self.model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy',f1])

        return self.model


    def trainModel(self,dataSet):
        from keras.callbacks import TensorBoard

        tensorboard = TensorBoard(log_dir="/tmp/zhangzyTFK",histogram_freq=1)

        print(self.name+" training")
        t0=time.time()
        counter=collections.Counter(dataSet.dataY)
        l1=float(counter[0])
        l2=float(counter[1])
        l=max(l1,l2)
        cls_w={0:100*l2/l,1:100*l1/l}
        print(counter)
        print("class weight",cls_w)

        self.buildModel()

        em1,em2 = dataSet.dataEm1,dataSet.dataEm2
        label = dataSet.dataY
        feeddata={"em1":em1,"em2":em2}
        feedlabel={"label":utils.to_categorical(label,2)}

        self.model.fit(feeddata,feedlabel,
            verbose=2, epochs=50, batch_size=500, class_weight=cls_w
                       #,callbacks=[tensorboard]
                       ,validation_split=0.2
                       )

        t1=time.time()

        print("finished in %ds"%(t1-t0))

    def predict(self,dataSet):
        em1, em2 = dataSet.dataEm1, dataSet.dataEm2
        feeddata = {"em1": em1, "em2": em2}

        Y=self.model.predict(feeddata,verbose=0)
        Y=np.argmax(Y,axis=1)

        print(self.name,"finished predicting %d records"%len(em1))
        return Y

    def loadModel(self):
        self.buildModel()
        self.model.load_weights("./models/" + self.name + ".h5")


        print("loaded",self.name,"model")

    def saveModel(self):
        self.model.save_weights("./models/" + self.name + ".h5")

        print("saved",self.name,"model")



if __name__ == '__main__':
    from WordModel import WordEmbedding
    from FeatureDataSet import NLPDataSet

    data = NLPDataSet(testMode=False)
    data.loadDocsData("../data/train_nlp_data.csv")
    docs = data.getAllDocs()

    #embedding words
    emModel = WordEmbedding()
    emModel.loadModel()

    embeddings=emModel.transformDoc2Vec(docs)


    n_count = len(embeddings)
    em1 = embeddings[:n_count // 2]
    em2 = embeddings[n_count // 2:]

    labels = np.array(data.docdata["label"], dtype=np.int)

    data.constructData(em1=em1,em2=em2,labels=labels)

    dnnmodel=TwoInDNNModel()

    dnnmodel.trainModel(data)
    dnnmodel.saveModel()
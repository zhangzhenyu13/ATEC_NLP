from keras import models,layers,optimizers,losses
import time,keras.backend as K
from keras import utils
from sklearn import metrics
<<<<<<< HEAD
import warnings
=======
import warnings,pickle
>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13
import numpy as np
import collections
import initConfig
from keras.callbacks import TensorBoard

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
<<<<<<< HEAD
=======
        self.params={

            'dp':0.8,
            'verbose':0,
        }
>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13

        self.tensorboard=None

    def buildModel(self):
        datashape=(initConfig.config["maxWords"],initConfig.config["features"])
        outputDim=2
        input1=layers.Input(shape=datashape)
        input2=layers.Input(shape=datashape)
        comLSTM=layers.LSTM(64)
        encode1=comLSTM(input1)
        encode2=comLSTM(input2)
        mergeLayer=layers.concatenate([encode1,encode2],axis=-1)
        hiddenLayer=layers.Dense(64,activation="relu")(mergeLayer)
<<<<<<< HEAD
        hiddenLayer = layers.Dense(64, activation="relu")(hiddenLayer)
        hiddenLayer=layers.GaussianDropout(0.7)(hiddenLayer)
=======
>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13
        predictionLayer=layers.Dense(units=outputDim,activation="tanh")(hiddenLayer)
        self.model=models.Model(inputs=[input1,input2],outputs=predictionLayer)
        self.model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy',f1])

<<<<<<< HEAD

=======
>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13
        return self.model

    def StartTensorBoard(self):
        self.tensorboard=TensorBoard(log_dir="/tmp/zhangzyTFK",histogram_freq=1)


    def trainModel(self,dataSet):
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

        trainX1,trainX2 = dataSet.dataX1,dataSet.dataX2
        trainY = dataSet.dataY

        self.model.fit([trainX1, trainX2], utils.to_categorical(trainY, 2),
<<<<<<< HEAD
            verbose=2, epochs=10, batch_size=1000, class_weight=cls_w,callbacks=[self.tensorboard],validation_split=0.2)
=======
            verbose=2, epochs=300, batch_size=1000, class_weight=cls_w,callbacks=[self.tensorboard],validation_split=0.2)
>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13


        t1=time.time()

        #test training error
        y_predict=self.predict([trainX1,trainX2])

        f1=metrics.f1_score(trainY,y_predict)

        acc=metrics.accuracy_score(trainY,y_predict)

        print("finished in %ds"%(t1-t0),"f1=",f1,"acc=",acc)

    def predict(self,X):

        Y=self.model.predict(X,verbose=0)
        Y=np.argmax(Y,axis=1)

        print(self.name,"finished predicting %d records"%len(X))
        return Y

    def loadModel(self):
<<<<<<< HEAD
        self.buildModel()
        self.model.load_weights("./models/" + self.name + ".h5")
=======

        self.model = models.load_model("./models/" + self.name + ".h5")
>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13

        print("loaded",self.name,"model")

    def saveModel(self):
<<<<<<< HEAD
        self.model.save_weights("./models/" + self.name + ".h5")
=======
        self.model.save("./models/" + self.name + ".h5")
>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13
        print("saved",self.name,"model")



if __name__ == '__main__':
    from WordModel import WordEmbedding
<<<<<<< HEAD
    from FeatureDataSet import WordDataSet

    data = WordDataSet(testMode=False)
    data.loadDocsData("./data/train_nlp_data.csv")
    docs = data.getAllDocs()

    docModel = WordEmbedding()
    docModel.loadModel()
=======
    from FeatureDataSet import DocDataSet

    docdata = DocDataSet(testMode=False)
    docdata.loadDocsData("./data/train_nlp_data.csv")
    docModel = WordEmbedding()
    docs = docdata.getAllDocs()
    docModel.trainDocModel(docs)

>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13
    embeddings=docModel.transformDoc2Vec(docs)

    n_count = len(embeddings)
    s1 = embeddings[:n_count // 2]
    s2 = embeddings[n_count // 2:]
<<<<<<< HEAD
    labels = np.array(data.docdata["label"], dtype=np.int)

    data.constructData(s1, s2, labels)

    dnnmodel=TwoInDNNModel()
    dnnmodel.StartTensorBoard()
    dnnmodel.trainModel(data)
    dnnmodel.saveModel()
=======
    labels = np.array(docdata.docdata["label"], dtype=np.int)

    docdata.constructData(s1, s2, labels)

    dnnmodel=TwoInDNNModel()
    dnnmodel.StartTensorBoard()
    dnnmodel.trainModel(docdata)
>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13

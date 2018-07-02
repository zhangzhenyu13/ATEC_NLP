from keras import models,layers,optimizers,losses,metrics
import time
from keras import utils
import warnings
import numpy as np
import initConfig
import keras.backend as K
warnings.filterwarnings("ignore")
from LSTMDNN import MyMetrics
#model
class CNNModel:

    def __init__(self):
        self.name="TwoInputCNN"

    def buildModel(self):
        datashape = (initConfig.config["maxWords"], initConfig.config["features"])

        #word net
        input1=layers.Input(shape=datashape,name="em1")
        input2=layers.Input(shape=datashape,name="em2")

        comCNN = layers.Conv1D(filters=32,kernel_size=5,padding="same",activation="relu")
        comPool = layers.AveragePooling1D(pool_size=5)

        comCNN2 = layers.Conv1D(filters=64, kernel_size=5,padding="same", activation="relu")
        comPool2 = layers.AveragePooling1D(pool_size=2)

        x1=comCNN(input1)
        x2=comCNN(input2)
        x1=comPool(x1)
        x2=comPool(x2)

        x1 = comCNN2(x1)
        x2 = comCNN2(x2)
        x1 = comPool2(x1)
        x2 = comPool2(x2)

        flatLayer=layers.Flatten()
        feature1=flatLayer(x1)
        feature2=flatLayer(x2)

        # sim net
        L1_distance = lambda x: K.abs(x[0] - x[1])
        both = layers.merge([feature1, feature2], mode=L1_distance, output_shape=lambda x: x[0])
        hiddenLayer=layers.Dense(units=1024,activation="relu")(both)
        dropLayer=layers.Dropout(0.5)(hiddenLayer)

        predictionLayer=layers.Dense(units=2,name="label",activation="softmax")(dropLayer)
        self.model=models.Model(inputs=[input1,input2],
                                outputs=[
                                    predictionLayer,
                                ]
                                )

        self.model.compile(optimizer=optimizers.Adam(),
                      loss={
                          "label":losses.binary_crossentropy
                           }
                      )

        return self.model


    def trainModel(self,train,validate):
        from keras.callbacks import TensorBoard
        import collections
        print(collections.Counter(train.dataY))
        tensorboard = TensorBoard(log_dir="/tmp/zhangzyTFK",histogram_freq=1)
        watch_metrics = MyMetrics()

        print(self.name+" training")
        t0=time.time()

        cls_w={0:1,1:4}

        print("class weight",cls_w)

        self.buildModel()

        em1,em2 = train.dataEm1,train.dataEm2
        label = train.dataY
        feeddata={"em1":em1,"em2":em2}
        feedlabel={
            "label":utils.to_categorical(label,2)
        }

        if validate is not None:

            val_em1, val_em2 = validate.dataEm1, validate.dataEm2
            val_label = validate.dataY
            val_feeddata = {"em1": val_em1, "em2": val_em2}
            val_feedlabel = {
                "label": utils.to_categorical(val_label, 2)
            }
            val_data=(val_feeddata,val_feedlabel)
        else:

            val_data=None

        self.model.fit(feeddata,feedlabel,
            verbose=2, epochs=8, batch_size=500

                       ,validation_data=val_data
                       ,class_weight={"label":cls_w}
                       ,callbacks=[tensorboard,watch_metrics]
                       #,validation_split=0.2
                       )

        t1=time.time()

        print("finished in %ds"%(t1-t0))

    def predict(self,dataSet):
        em1, em2 = dataSet.dataEm1, dataSet.dataEm2
        feeddata = {"em1": em1, "em2": em2}

        Y=self.model.predict(feeddata,verbose=0)
        #print(Y)
        #Y=Y[0]
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


def getFeedData(dataPath):
    data = NLPDataSet(testMode=False)
    data.loadDocsData(dataPath)
    docs = data.getAllDocs()

    embeddings = emModel.transformDoc2Vec(docs)

    n_count = len(embeddings)
    em1 = embeddings[:n_count // 2]
    em2 = embeddings[n_count // 2:]

    labels = np.array(data.docdata["label"], dtype=np.int)

    data.constructData(em1=em1, em2=em2, labels=labels)

    return data

if __name__ == '__main__':
    from FeatureDataSet import NLPDataSet
    from utilityFiles import splitTrainValidate
    from WordModel import WordEmbedding
    # embedding words
    emModel = WordEmbedding()
    emModel.loadModel()

    #lstm dnn model

    splitratio=1
    if splitratio>0 and splitratio<1:
        splitTrainValidate("../data/train_nlp_data.csv",splitratio)

        trainData=getFeedData("../data/train.csv")
        validateData=getFeedData("../data/validate.csv")
    else:
        trainData,validateData=getFeedData("../data/train_nlp_data.csv"),None

    model_num = initConfig.config["cnnNum"]
    dataList = trainData.getFold(model_num)
    for i in range(model_num):
        dnnmodel = CNNModel()

        dnnmodel.name += str(i)
        train, test = dataList[i]
        dnnmodel.trainModel(train, test)
        dnnmodel.saveModel()

        print("\n===========================\n")




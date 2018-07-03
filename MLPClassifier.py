from DNNModel import *

class MLPModel(TwoInDNNModel):
    def __init__(self):
        TwoInDNNModel.__init__(self)
        self.name="TwoInputMLP"

    def buildModel(self):
        datashape = (initConfig.config["docLen"],)

        # word net
        input1 = layers.Input(shape=datashape, name="s1")
        input2 = layers.Input(shape=datashape, name="s2")

        comL1=layers.Dense(units=196,activation="relu")
        comL2=layers.Dense(units=128,activation="relu")
        comL3=layers.Dense(units=96,activation="relu")

        x1=comL1(input1)
        x1=comL2(x1)
        x1=comL3(x1)

        x2=comL1(input2)
        x2=comL2(x2)
        x2=comL3(x2)

        # sim net
        L1_distance = lambda x: K.abs(x[0] - x[1])
        both = layers.merge([x1, x2], mode=L1_distance, output_shape=lambda x: x[0])
        hiddenLayer = layers.Dense(units=64, activation="relu", bias_initializer=b_init)(both)

        dropLayer = layers.Dropout(0.5)(hiddenLayer)

        predictionLayer = layers.Dense(units=2, name="label", activation="softmax")(dropLayer)
        self.model = models.Model(inputs=[input1, input2],
                                  outputs=[
                                      predictionLayer,
                                  ]
                                  )

        self.model.compile(optimizer=optimizers.Adam(),
                           loss={
                               "label": losses.binary_crossentropy
                           }
                           )

        return self.model


def getFeedData(dataPath,emModel):
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
    from DocModel import DocVectorizer
    # embedding words
    emModel = DocVectorizer()
    emModel.loadModel()

    splitratio=1
    if splitratio>0 and splitratio<1:
        splitTrainValidate("../data/train_nlp_data.csv",splitratio)

        trainData=getFeedData("../data/train.csv",emModel)
        validateData=getFeedData("../data/validate.csv",emModel)
        dnnmodel = MLPModel()

        dnnmodel.trainModel(trainData, validateData)
        dnnmodel.saveModel()
        exit(1)
    else:
        trainData,validateData=getFeedData("../data/train_nlp_data.csv",emModel),None

    model_num = initConfig.config["cnnNum"]
    dataList = trainData.getFold(model_num)
    for i in range(model_num):
        # cnn dnn model
        dnnmodel = MLPModel()

        dnnmodel.name += str(i)
        train, test = dataList[i]
        dnnmodel.trainModel(train, test)
        dnnmodel.saveModel()

        print("\n==========%d/%d=================\n"%(i+1,model_num))

    exit(2)

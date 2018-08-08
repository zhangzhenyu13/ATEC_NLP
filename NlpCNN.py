from keras import models,layers,optimizers,losses,metrics
import time
from keras import utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import warnings
import numpy as np
import pandas as pd
from sklearn import metrics
from keras.callbacks import Callback
import keras.backend as K
import pickle,jieba
from keras.applications import resnet50
warnings.filterwarnings("ignore")

class NLPDataSet:

    def __init__(self,testMode):

        self.docdata=None
        self.testMode=testMode

    def loadDicts(self):

        with open("./data/userdict-ch.txt","r") as f:
            for w in f:
                jieba.add_word(w.replace("\r","").replace("\n",""))

        self.stopwords=set()
        with open("./data/stopwords-ch.txt","r") as f:
            for w in f:
                self.stopwords.add(w.replace("\r","").replace("\n",""))

    def loadDocsData(self):

        inputfile="./data/train_nlp_data.csv"
        with open(inputfile,"r") as f:
            records=[]
            for line in f:
                record=line.replace("\n","").replace("\r", "").split("\t")
                records.append(record)
        if self.testMode:
            self.docdata=pd.DataFrame(data=records,columns=["id","sent1","sent2"])
        else:
            self.docdata=pd.DataFrame(data=records,columns=["id","sent1","sent2","label"])

        self.loadDicts()
        cleanVec=np.vectorize(self.cleanDoc)
        self.docdata["sent1"]=cleanVec(self.docdata["sent1"])
        self.docdata["sent2"]=cleanVec(self.docdata["sent2"])

        print("loaded %d records"%len(self.docdata["id"]))

    def cleanDoc(self,doc):

        cut_words=jieba.cut_for_search(doc)
        words=[]
        for word in cut_words:
            if word in self.stopwords:
                continue
            words.append(word)
        words=" ".join(words)

        return words

    def constructData(self):
        """

        :param s1:
        :param s2:
        :param labels:
        :return:
        """
        s1, s2 = self.docdata["sent1"], self.docdata["sent2"]
        self.text1=s1
        self.text2=s2
        if self.testMode:
            self.label=np.zeros(shape=len(s1),dtype=np.int)
        else:
            self.label = np.array(self.docdata["label"],dtype=np.int)

    def getFold(self,fold_num=5):

        from sklearn import model_selection
        kfold=model_selection.KFold(n_splits=fold_num,shuffle=True)
        dataList=[]
        Y=np.reshape(self.label,newshape=(len(self.label),1))

        for train_index, test_index in kfold.split(Y):

            data_train=NLPDataSet(False)
            data_test=NLPDataSet(False)

            data_train.text1,data_train.text2,data_train.label=\
                self.text1[train_index],self.text2[train_index],self.label[train_index]
            data_test.text1,data_test.text2,data_test.label=\
                self.text1[test_index],self.text2[test_index],self.label[test_index]

            dataList.append([data_train,data_test])

        return dataList


    def getInitialFold(self,fold_num=5):

        from sklearn import model_selection
        kfold = model_selection.KFold(n_splits=fold_num, shuffle=True)
        dataList = []
        Y=np.array(self.docdata["label"])
        Y = np.reshape(Y, newshape=(len(Y), 1))
        for train_index, test_index in kfold.split(Y):
            data_train=NLPDataSet(False)
            data_test=NLPDataSet(False)
            data_train.docdata=self.docdata.loc[train_index]
            data_test.docdata=self.docdata.loc[test_index]

            dataList.append([data_train,data_test])

        return dataList

class MyMetrics(Callback):
    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_cms=[]
        self.val_acc=[]
    def on_epoch_end(self, epoch, logs=None):

        '''
        print(type(self.validation_data))
        for i in range(len(self.validation_data)):
            print(np.shape(self.validation_data[i]))

        print("show data")
        for i in range(2,len(self.validation_data)):
            print(self.validation_data[i][:10])
        '''


        feed_data = {"seq1": self.validation_data[0], "seq2": self.validation_data[1]}
        predicts=self.model.predict(feed_data)#[0]
        val_predict = np.argmax(predicts,axis=1)
        val_targ = np.argmax(self.validation_data[2],axis=1)

        #metrics print
        CM=metrics.confusion_matrix(val_targ,val_predict)
        tn, fp, fn, tp = CM.ravel()
        if tp+fp==0:
            precision=0
        else:
            precision = float(tp) / float(tp + fp)
        if fn+tp==0:
            recall=0
        else:
            recall = float(tp) / float(fn + tp)

        print("tp,fp,tn,fn", tp, fp, tn, fn)
        print(CM)
        print("precision,recall", precision, recall)
        # measure
        f1 = 2 * precision * recall / (precision + recall)

        print("f1-score=", f1)

        acc = float(tp + tn) / float(tn + fp + fn + tp)

        print("accuracy=", acc)

        print()
        return

#model
def buildTokenizer(tokenModel,data):
        #tkonizer model
        t0=time.time()
        text1,text2=data.text1,data.text2
        texts=np.concatenate((text1,text2),axis=0)
        tokenModel.fit_on_texts(texts)
        t1=time.time()
        print("tokenizer building finished in %ds"%(t1-t0))




from keras.layers import Activation,Flatten,Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import BatchNormalization,ZeroPadding2D


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters

    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_tensor=None):

    bn_axis = 3

    img_input = input_tensor

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = BatchNormalization( axis=bn_axis,name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    '''
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    '''

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)



    return x

class NlpModel:
    tokenModel=Tokenizer(num_words=10000)
    Size_Vocab=10000

    def __init__(self):
        self.name="NlpCNNDecisionModel"
        self.model=None
        self.numEpoch=10
        self.batch_size=64
        self.maxLen=128
        self.embeddingSize=196
        self.class_weight={0:1,1:4.5}

        if "model_dir" in vars():
            self.model_dir=model_dir
        else:
            self.model_dir="./models/"


    def buildModel(self):

        #word net
        input1=layers.Input(shape=(self.maxLen,),name="seq1")
        input2=layers.Input(shape=(self.maxLen,),name="seq2")

        comEmbedding=layers.Embedding(input_dim=self.Size_Vocab,output_dim=self.embeddingSize,input_length=self.maxLen)
        emb1=comEmbedding(input1)
        emb2=comEmbedding(input2)

        comReshape=layers.Reshape((self.maxLen,self.embeddingSize,1))
        pic1=comReshape(emb1)
        pic2=comReshape(emb2)
        pic3=layers.subtract([pic1,pic2])
        pics=layers.Concatenate(axis=2)([pic1,pic2,pic3])
        #print(K.ndim(pics))
        #cnn
        x=ResNet50(pics)
        # extract net2
        x=layers.Dropout(0.5)(x)
        predictionLayer=layers.Dense(units=2,name="label",activation="softmax")(x)
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
        print(collections.Counter(train.label))
        tensorboard = TensorBoard(log_dir="/tmp/zhangzyTFK",histogram_freq=1)
        watch_metrics = MyMetrics()
        print(self.name+" training")
        print("class weight",self.class_weight)

        #siamese networks
        t0=time.time()

        self.buildModel()

        label = train.label
        text1,text2=train.text1,train.text2
        seq1=self.tokenModel.texts_to_sequences(text1)
        seq2=self.tokenModel.texts_to_sequences(text2)
        seq1,seq2=sequence.pad_sequences(seq1,self.maxLen),sequence.pad_sequences(seq2,self.maxLen)
        feeddata={"seq1":seq1,"seq2":seq2}
        feedlabel={
            "label":utils.to_categorical(label,2)
        }

        if validate is not None:

            vtext1,vtext2=validate.text1,validate.text2
            vseq1,vseq2=self.tokenModel.texts_to_sequences(vtext1),self.tokenModel.texts_to_sequences(vtext2)
            vseq1,vseq2=sequence.pad_sequences(vseq1,self.maxLen),sequence.pad_sequences(vseq2,self.maxLen)
            val_label = validate.label
            val_feeddata = {"seq1":vseq1,"seq2":vseq2}
            val_feedlabel = {
                "label": utils.to_categorical(val_label, 2)
            }
            val_data=(val_feeddata,val_feedlabel)
        else:

            val_data=None


        self.model.fit(feeddata,feedlabel,
            verbose=1, epochs=self.numEpoch, batch_size=self.batch_size

                       ,validation_data=val_data
                       ,class_weight={"label":self.class_weight}
                       ,callbacks=[watch_metrics]
                       #,validation_split=0.2
                       )

        t1=time.time()

        print("siamese model(%s) training finished in %ds"%(self.name,t1-t0))

        #print(self.model.get_weights())

    def predict(self,dataSet):
        text1,text2=dataSet.text1, dataSet.text2
        seq1=self.tokenModel.texts_to_sequences(text1)
        seq2=self.tokenModel.texts_to_sequences(text2)
        seq1,seq2=sequence.pad_sequences(seq1,self.maxLen),sequence.pad_sequences(seq2,self.maxLen)
        feeddata = {"seq1": seq1, "seq2": seq2}

        Y=self.model.predict(feeddata,verbose=0)

        Y=np.argmax(Y,axis=1)

        print(self.name,"finished predicting %d records"%len(text1))
        return Y

    def loadModel(self):
        self.model=models.load_model(self.model_dir + self.name + ".h5")
        with open(self.model_dir+"tokneModel","rb") as f:
            self.tokenModel=pickle.load(f)

        print("loaded",self.name,"model")

    def saveModel(self):
        self.model.save(self.model_dir + self.name + ".h5")
        with open(self.model_dir+"tokneModel","wb") as f:
            pickle.dump(self.tokenModel,f)

        print("saved",self.name,"model")


def trainModel(MyModel):

    data=NLPDataSet(False)
    data.loadDocsData()
    data.constructData()
    buildTokenizer(MyModel.tokenModel,data)
    dataList=data.getFold(model_num)

    for i in range(model_num):
        if train_index!=i :continue

        # lstm dnn model
        dnnmodel = MyModel()

        dnnmodel.name += str(i)
        train, test = dataList[i]
        dnnmodel.trainModel(train, test)
        dnnmodel.saveModel()

        print("\n==========%d/%d=================\n" % (i + 1, model_num))

if __name__ == '__main__':
    model_num = 5
    train_index=0
    trainModel(NlpModel)

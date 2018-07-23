from keras import models,layers,optimizers,losses
import keras.backend as K
import numpy as np


#ini func
def W_init(shape, name=None):
    """Initialize weights as in paper"""
    values = np.random.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)

def b_init(shape, name=None):
    """Initialize bias as in paper"""
    values = np.random.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)

def modelStructDef(inpiutDim,outputDim):
    inputLayer=layers.Input(shape=(inpiutDim,))

    x=layers.Dense(64,activation="relu",bias_initializer=b_init)(inputLayer)

    x=layers.Reshape(target_shape=(8,8))(x)

    x=layers.LSTM(8,kernel_initializer=W_init)(x)

    y=layers.Dense(outputDim,name="label")(x)

    model=models.Model(inputs=[inputLayer],outputs=[y])

    return model

def compileModel(model):
    model.compile(optimizer=optimizers.Adam(),
                      loss={
                          "label":losses.binary_crossentropy
                           }
                      )
    return model
def loadModel(struct_json):
    return models.model_from_config(struct_json)

if __name__ == '__main__':
    X=np.random.random((1000,20))
    Y=np.zeros(shape=(1000,2),dtype=np.int)
    import random
    for i in range(1000):
        pos=random.randint(0,1)
        #print(i,pos)
        Y[i][pos]=0
    model=modelStructDef(20,2)

    model=compileModel(model)

    model.fit(X,Y)

    struct=model.get_config()
    print("model struct")
    print(struct)

    model1=loadModel(struct)

    model1=compileModel(model1)

    print(model1.to_json())

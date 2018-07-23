from DNNModel import *
from keras.layers import (Conv2D,Activation,BatchNormalization,Input,GlobalMaxPooling2D,Dense,Flatten,
                          ZeroPadding2D,MaxPooling2D,AveragePooling2D,GlobalAveragePooling2D)

class ResNetModel(TwoInDNNModel):
    def __init__(self):
        TwoInDNNModel.__init__(self)
        self.name = "TwoInputResNet"

    def initLayers(self):
        self.AllLayers={

        }

    def storeLayer(self,name,alayer):
        if name not in self.AllLayers.keys():
            self.AllLayers[name]=alayer

    def identity_block(self,input_tensor, kernel_size, filters, stage, block):
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

        name=conv_name_base + '2a'
        self.storeLayer(name=name,alayer=Conv2D(filters1, (1, 1), name=conv_name_base + '2a'))
        x = self.AllLayers[name](input_tensor)

        name = bn_name_base + '2a'
        self.storeLayer(name=name,alayer=BatchNormalization(axis=bn_axis, name=bn_name_base + '2a'))
        x = self.AllLayers[name](x)

        name += "afterActivation"
        self.storeLayer(name=name,alayer=Activation('relu'))
        x = self.AllLayers[name](x)

        name = conv_name_base + '2b'
        self.storeLayer(name=name,alayer=Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b'))
        x = self.AllLayers[name](x)

        name = bn_name_base + '2b'
        self.storeLayer(name=name,alayer=BatchNormalization(axis=bn_axis, name=bn_name_base + '2b'))
        x = self.AllLayers[name](x)

        name += "afterActivation"
        self.storeLayer(name=name,alayer=Activation('relu'))
        x = self.AllLayers[name](x)

        name = conv_name_base + '2c'
        self.storeLayer(name=name,alayer=Conv2D(filters3, (1, 1), name=conv_name_base + '2c'))
        x = self.AllLayers[name](x)

        name=bn_name_base + '2c'
        self.storeLayer(name=name,alayer=BatchNormalization(axis=bn_axis, name=bn_name_base + '2c'))
        x = self.AllLayers[name](x)

        name += "add"
        self.storeLayer(name=name, alayer=layers.Add())
        x = self.AllLayers[name]([x, input_tensor])

        name += "afterActivation"
        self.storeLayer(name=name,alayer=Activation('relu'))
        x = self.AllLayers[name](x)
        return x

    def conv_block(self,input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
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

        name = conv_name_base + '2a'
        self.storeLayer(name=name,alayer=Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a'))
        x = self.AllLayers[name](input_tensor)

        name = bn_name_base + '2a'
        self.storeLayer(name=name,alayer=BatchNormalization(axis=bn_axis, name=bn_name_base + '2a'))
        x = self.AllLayers[name](x)

        name += "afterActivation"
        self.storeLayer(name=name,alayer=Activation('relu'))
        x = self.AllLayers[name](x)

        name = conv_name_base + '2b'
        self.storeLayer(name=name,alayer=Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b'))
        x = self.AllLayers[name](x)

        name = bn_name_base + '2b'
        self.storeLayer(name=name,alayer=BatchNormalization(axis=bn_axis, name=bn_name_base + '2b'))
        x = self.AllLayers[name](x)


        name += "afterActivation"
        self.storeLayer(name=name,alayer=Activation('relu'))
        x = self.AllLayers[name](x)

        name = conv_name_base + '2c'
        self.storeLayer(name=name,alayer=Conv2D(filters3, (1, 1), name=conv_name_base + '2c'))
        x = self.AllLayers[name](x)

        name = bn_name_base + '2c'
        self.storeLayer(name=name,alayer=BatchNormalization(axis=bn_axis, name=bn_name_base + '2c'))
        x = self.AllLayers[name](x)

        name = conv_name_base + '1'
        self.storeLayer(name=name,alayer=Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1'))
        shortcut = self.AllLayers[name](input_tensor)

        name = bn_name_base + '1'
        self.storeLayer(name=name,alayer=BatchNormalization(axis=bn_axis, name=bn_name_base + '1'))
        shortcut = self.AllLayers[name](shortcut)

        name +="add"
        self.storeLayer(name=name,alayer=layers.Add())
        x = self.AllLayers[name]([x, shortcut])

        name += "afterActivation"
        self.storeLayer(name=name,alayer=Activation('relu'))
        x = self.AllLayers[name](x)

        return x

    def ResNet50(self,input):

        conv_block,identity_block=self.conv_block,self.identity_block

        img_input = input
        bn_axis=3

        name = 'conv1_pad'
        self.storeLayer(name=name,alayer=ZeroPadding2D(padding=(3, 3), name='conv1_pad'))
        x = self.AllLayers[name](img_input)

        name = 'conv1'
        self.storeLayer(name=name,alayer=Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1'))
        x = self.AllLayers[name](x)

        name = 'bn_conv1'
        self.storeLayer(name=name, alayer=BatchNormalization(axis=bn_axis, name='bn_conv1'))
        x = self.AllLayers[name](x)

        name += "afterActivation"
        self.storeLayer(name=name, alayer=Activation('relu'))
        x = self.AllLayers[name](x)

        name += "afterPool"
        self.storeLayer(name=name, alayer=MaxPooling2D((3, 3), strides=(2, 2)))
        x = self.AllLayers[name](x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        #x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        #x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        #x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        #x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        #x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        #x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        #x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        #x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        #x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        #x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        #x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        #x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        #x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        name="avgPool"
        self.storeLayer(name=name,alayer=AveragePooling2D((2, 2), name='avg_pool'))
        x = self.AllLayers[name](x)

        name="Faltten"
        self.storeLayer(name=name,alayer=Flatten())
        x=self.AllLayers[name](x)

        return x

    def buildModel(self):
        self.initLayers()

        datashape = (initConfig.config["maxWords"], initConfig.config["features"])
        # word net
        input1 = Input(shape=datashape, name="em1")
        input2 = Input(shape=datashape, name="em2")

        newshape = (initConfig.config["maxWords"], initConfig.config["features"], 1)
        reshLayer = layers.Reshape(target_shape=newshape)
        x1 = reshLayer(input1)
        x2 = reshLayer(input2)

        x1=self.ResNet50(x1)
        x2=self.ResNet50(x2)


        L1_distance = lambda x: K.abs(x[0] - x[1])
        both = layers.merge([x1, x2], mode=L1_distance, output_shape=lambda x: x[0])

        hiddenLayer = layers.Dense(units=64, activation="relu", bias_initializer=b_init)(both)

        dropLayer = layers.Dropout(0.36)(hiddenLayer)

        predictionLayer = layers.Dense(units=2, name="label", activation="softmax")(dropLayer)
        self.model = models.Model(inputs=[input1, input2],
                                  outputs=[
                                      predictionLayer,
                                  ]
                                  , name='resnet50'
                                  )

        self.model.compile(optimizer=optimizers.Adam(),
                           loss={
                               "label": losses.binary_crossentropy
                           }
                           )

        return self.model


if __name__ == '__main__':
    trainModel(ResNetModel)

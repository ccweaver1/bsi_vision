try:
    from keras.applications import resnet50, vgg16, inception_v3
    from keras.models import Model
    from keras.layers import Input, Dense, Flatten, Dropout, ZeroPadding3D, \
                    GlobalAveragePooling2D, TimeDistributed, Flatten, ZeroPadding2D, \
                    Convolution2D, MaxPooling2D
except ImportError:
    from tensorflow.python.keras.applications import resnet50, vgg16, inception_v3
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.layers import Input, Dense, Flatten, Dropout, \
                    ZeroPadding3D, GlobalAveragePooling2D, TimeDistributed, Flatten, \
                    ZeroPadding2D, Convolution2D, MaxPooling2D

# from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)

class ResearchModels:
    
    def __init__(self, config, load_imagenet=True):
        model_name = config['model']['model_name']
        nb_outputs = config['model']['nb_outputs']
        input_shape = tuple(config['model']['target_shape'])
        if len(input_shape) == 2:
            input_shape = input_shape + (3,)

        if model_name ==  'resnet50_pool':
            print("Loading ResNet50 + GlobalMaxPooling Model")
            self.model = self.resnet50_pool(load_imagenet, nb_outputs, input_shape)
        elif model_name == 'vgg16_pool':
            print("Loading VGG16 + GlobalMaxPooling Model")
            self.model = self.vgg16_pool(nb_outputs, input_shape, load_imagenet)
        elif model_name == 'resnet50_pool_dense':
            self.model = self.resnet50_pool_dense(weight, nb_outputs, input_shape)
        elif model_name == 'dense':
            self.model = self.dense(nb_outputs, input_shape)
        elif model_name == 'inception_pool':
            print("Loading InceptionV3")
            self.model = self.inception_pool(nb_outputs, input_shape, load_imagenet)

        # print(self.model.summary())

    def resnet50_pool(self, load_imagenet, nb_outputs, input_shape):
        
        input_layer = Input(shape=input_shape)
        if load_imagenet:
            base_model = resnet50.ResNet50(weights='imagenet',
                                            input_tensor=input_layer,
                                            pooling='max',
                                            include_top=False)
        else:
            base_model = resnet50.ResNet50(include_top=False,
                                            input_tensor=input_layer,
                                            pooling='max')
        x = base_model.output
        # print x.output
        # x = Flatten()(x)
        predictions = Dense(nb_outputs, activation='linear', use_bias=False, kernel_initializer='zero', name='m_matrix_regr_out')(x)
        # x = TimeDistributed(Flatten())(x)
        # predictions = TimeDistributed(Dense(nb_outputs, activation='linear', use_bias=False, kernel_initializer='zero', name='m_matrix_regr_out'))(x)
        # predictions = tf.reshape(predictions, [-1])
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def resnet50_pool_dense(self, load_imagenet, nb_outputs, input_shape):
        input_layer = Input(shape=(None,) + input_shape)
        if load_imagenet:
            base_model = resnet50.ResNet50(weights='imagenet',
                                            input_tensor=input_layer,
                                            include_top=False)
        else:
            base_model = resnet50.ResNet50(include_top=False,
                                            input_tensor=input_layer)
        x = base_model.output
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(nb_outputs, activation='linear', kernel_initializer='zero', name='m_matrix_regr_out')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def dense(self, nb_outputs, input_shape):
        # input_shape = (None, None, 3)
        img_input = Input(shape=input_shape)

        # model = Sequential()
        # model.add(input_layer)
        x = ZeroPadding2D((3, 3))(img_input)
        x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', activation='relu')(x)
        x = Dense(10, activation='relu')(x)
        x = MaxPooling2D((7,7), strides=(5,5))(x)
        x = Flatten()(x)
        predictions = Dense(9, activation='linear', kernel_initializer='zero')(x)
        model = Model(inputs=img_input, outputs=predictions)
        
        return model

    def vgg16_pool(self, nb_outputs, input_shape, load_imagenet=True):
        input_layer = Input(shape=input_shape)
        if load_imagenet:
            base_model = vgg16.VGG16(weights='imagenet',
                                            input_tensor=input_layer,
                                            pooling='max',
                                            include_top=False)
        else:
            base_model = vgg16.VGG16(include_top=False,
                                            input_tensor=input_layer,
                                            pooling='max')
        x = base_model.output
        # print x.output
        # x = Flatten()(x)
        predictions = Dense(nb_outputs, activation='linear', use_bias=False, kernel_initializer='zero', name='m_matrix_regr_out')(x)
        # x = TimeDistributed(Flatten())(x)
        # predictions = TimeDistributed(Dense(nb_outputs, activation='linear', use_bias=False, kernel_initializer='zero', name='m_matrix_regr_out'))(x)
        # predictions = tf.reshape(predictions, [-1])
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def inception_pool(self, nb_outputs, input_shape, load_imagenet=True):
        input_layer = Input(shape=input_shape)
        if load_imagenet:
            base_model = inception_v3.InceptionV3(weights='imagenet',
                                            input_tensor=input_layer,
                                            pooling='max',
                                            include_top=False)
        else:
            base_model = inception_v3.InceptionV3(include_top=False,
                                            input_tensor=input_layer,
                                            pooling='max')
        x = base_model.output

        predictions = Dense(nb_outputs, activation='linear', use_bias=False, kernel_initializer='zero', name='m_matrix_regr_out')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def get_model(self):
        # print self.model.summary()
        return self.model


if __name__ == '__main__':
    from config import ConfigManager
    cm = ConfigManager('inception_pool_config_sage.json')
    cfg = cm.get_json()
    rm = ResearchModels(cfg['training'], load_imagenet=False)
    model = rm.get_model()
    print model.summary()

# # Then remove the top so we get features not predictions.
# # From: https://github.com/fchollet/keras/issues/2371
# self.model.layers.pop()
# self.model.layers.pop()  # two pops to get to pool layer
# self.model.outputs = [self.model.layers[-1].output]
# self.model.output_layers = [self.model.layers[-1]]
# self.model.layers[-1].outbound_nodes = []

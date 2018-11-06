import numpy as np
from PIL import Image
from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
import cv2
import tensorflow as tf

mapping = {
    0: 'baroque',
    1: 'camo',
    2: 'color_block',
    3: 'leopard',
    4: 'solid_color'
}

class CNN(object):
    def __init__(self):
        self.create_model()
        self.graph = tf.get_default_graph()

    def create_model(self, input_shape=(224, 224, 3), output_classes=5):
        # optimizer, learn_rate = get_optimizer(optimizer, learn_rate, decay, momentum)

        base_model = applications.VGG16(weights=None, include_top=False,
                                        input_shape=input_shape)
        model_inputs = base_model.input
        common_inputs = base_model.output
        x = Flatten()(common_inputs)
        x = Dense(256, activation='tanh')(x)
        x = Dropout(0.3)(x)
        predictions_class = Dense(output_classes, activation='softmax',
                                  name='predictions_class')(x)

        model = Model(inputs=model_inputs, outputs=predictions_class)

        model.load_weights("best-weights-073.hdf5", by_name=True)
        self.model = model
        # return model

    def preprocess_image(self, img_path, aug=True, img_width=224, img_height=224):
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize((img_width, img_height))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def predict(self, img_path, input_shape=(224, 224, 3)):
        img = self.preprocess_image(img_path)
        img = np.resize(img, (1, 224, 224, 3))
        # K.clear_session()
        # model = create_model((224, 224, 3), 5)
        # output = self.model.predict(img)
        with self.graph.as_default():
            predict = np.argmax(self.model.predict(img)[0])
        return mapping[predict]
from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import tensorflow as tf
import numpy as np
import os


def load_preprocessed_data(features_path=None, labels_path=None):
    if features_path is None or labels_path is None:
        features_path = 'data/preprocessed/features.bin.npy'
        labels_path = 'data/preprocessed/names.bin.npy'
    features = np.load(features_path)
    features = np.reshape(features, (features.shape[0], features.shape[1]))
    labels = np.load(labels_path)
    return features, labels


def split_train_test(x, y, rate=0.8):
    num_train = int(len(x) * rate)
    x_train = x[:num_train]
    y_train = y[:num_train]

    x_test = x[num_train:]
    y_test = y[num_train:]
    return x_train, y_train, x_test, y_test


class PretrainedModel(object):
    def __init__(self):
        self.model = self.__load_model__()
        self.graph = tf.get_default_graph()

    def __load_model__(self):
        model = vgg16.VGG16(weights='imagenet', include_top=True)
        model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)
        return model_extractfeatures

    def get_extracted_feature(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        with self.graph.as_default():
            feature = self.model.predict(x).reshape(-1)
        # print(feature.shape)
        return feature

    def get_extracted_features(self, folder, model=None):
        path_labels = [os.path.join(folder, f) for f in os.listdir(folder)]
        path_img = []
        labels = []
        for path_label in path_labels:
            list_imgs = [os.path.join(path_label, f) for f in os.listdir(path_label)]
            for img in list_imgs:
                path_img.append(img)
                labels.append(path_labels.index(path_label))

        labels = np.asarray(labels)
        path_img = np.asarray(path_img)
        idx = np.random.permutation(len(path_img))

        labels = labels[idx]
        path_img = path_img[idx]

        features = []
        for idx, img_path in enumerate(path_img):
            print('image :', idx)
            features.append(self.get_extracted_feature(img_path))

        features = np.array(features)
        labels = np.array(labels)
        np.save('data/preprocessed/features1.npy', features)
        np.save('data/preprocessed/labels1.npy', labels)



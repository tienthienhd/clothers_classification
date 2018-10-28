import numpy as np
from sklearn.svm import *
import pickle
from preprocessing import *
from sklearn.model_selection import *

# class ModelSVM(object):
#     def __init__(self, kernel='linear', C=1, gamma='auto'):
#         self.model_encoder = None
#         self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma)
#
#     def train(self, x, y):
#         print('Start training...')
#         self.model.fit(x, y)
#         self.save_model('model_saved/svm.model')
#         print('Finish training !!!')
#         print(self.model.get_params())
#
#     def validate(self, x, y):
#         accuracy = self.model.score(x, y)
#         print('Accuracy=', accuracy)
#         return accuracy
#
#     def fit(self, x_train, y_train, x_test, y_test):
#         self.train(x_train, y_train)
#         accuracy = self.validate(x_test, y_test)
#         print('accuracy =', accuracy)
#
#     def predict(self, img_preprocessed=None, image_path=None):
#         if img_preprocessed is None and image_path is None:
#             return None
#         elif image_path is None:
#             return self.model.predict(img_preprocessed)
#         else:
#             if self.model_encoder is None:
#                 self.model_encoder = PretrainedModel()
#             img_preprocessed = self.model_encoder.get_extracted_feature(image_path)
#             return self.model.predict(img_preprocessed)
#
#     def save_model(self, filename):
#         pickle.dump(self.model, open(filename, 'wb'))
#
#     def load_model(self, filename):
#         self.model = pickle.load(open(filename, 'rb'))




features, labels = load_preprocessed_data()
x_train, y_train, x_test, y_test = split_train_test(features, labels, rate=0.9)

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
kernels = ["linear", "poly", "rbf", "sigmoid"]
param_grid = dict(gamma=gamma_range, C=C_range, kernel=kernels)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, n_jobs=-1)
grid.fit(x_train, y_train)
pickle.dump(grid, 'model_saved/grid_svm.model')

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
from preprocessing import *
from sklearn.model_selection import *


features, labels = load_preprocessed_data()
x_train, y_train, x_test, y_test = split_train_test(features, labels, rate=0.9)


def tuning():
    n_neighbors = range(5, 31)
    weights = ['distance']
    p = [2]
    param_grid = dict(n_neighbors=n_neighbors, weights=weights, p=p)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, n_jobs=-1, verbose=2)
    grid.fit(x_train, y_train)
    pickle.dump(grid, open('model_saved/grid_knn.model', 'wb'))


def retrain_best():
    print('retrain best param')
    grid = pickle.load(open('model_saved/grid_knn.model', 'rb'))
    params = grid.best_params_
    # print(grid.cv_results_)
    n_neighbors = params['n_neighbors']
    weights = params['weights']
    p = params['p']
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    knn.fit(x_train, y_train)
    pickle.dump(knn, open('model_saved/knn.model', 'wb'))
    print('train complete! ')

# tuning()
retrain_best()
knn_model = pickle.load(open('model_saved/knn.model', 'rb'))
acc = knn_model.score(x_test, y_test)
print(acc)

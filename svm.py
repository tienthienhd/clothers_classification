import numpy as np
from sklearn.svm import *
import pickle
from preprocessing import *
from sklearn.model_selection import *


features, labels = load_preprocessed_data()
x_train, y_train, x_test, y_test = split_train_test(features, labels, rate=0.9)


def tuning():
    C_range = np.logspace(-1, 3, 5)
    # gamma_range = np.logspace(-3, 1, 5)
    kernels = ["linear", "poly", "rbf"]
    param_grid = dict(gamma=['auto'], C=C_range, kernel=kernels)
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, n_jobs=-1, verbose=2)
    grid.fit(x_train, y_train)
    pickle.dump(grid, open('model_saved/grid_svm.model','wb'))

def retrain_best():
    print('retrain best param')
    grid = pickle.load(open('model_saved/grid_svm.model', 'rb'))
    params = grid.best_params_
    # print(grid.cv_results_)
    C = params["C"]
    kernel = params["kernel"]
    gamma = params["gamma"]
    svm = SVC(C=C, kernel=kernel, gamma=gamma)
    svm.fit(x_train, y_train)
    pickle.dump(svm, open('model_saved/svm.model', 'wb'))
    print('train complete! ')


retrain_best()
svm_model = pickle.load(open('model_saved/svm.model', 'rb'))
acc = svm_model.score(x_test, y_test)
print(acc)

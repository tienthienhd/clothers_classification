import matplotlib.pyplot as plt
import pandas as pd
import pickle

grid_knn = pickle.load(open('model_saved/grid_svm.model', 'rb'))
best_params = grid_knn.best_params_
del best_params['gamma']
# del best_params['weights']

cv_results = grid_knn.cv_results_
# df_results = pd.DataFrame(cv_results)

print(cv_results.keys())
mean_test_score = cv_results['mean_test_score']
params = cv_results['params']
print(params)

params = [str(a['C']) + ',' + str(a['kernel']) for a in params]
print(params)
plt.plot(params, mean_test_score)
plt.xticks(params, rotation='vertical')
plt.xlabel('parameter (C, kernel)')
plt.ylabel('accuracy')
plt.title('Accuracy grid search SVM with best param:'+ str(best_params))
plt.show()

# params = [a['n_neighbors'] for a in params]
# print(params)
# plt.plot(params, mean_test_score)
# # plt.xticks(params, rotation='vertical')
# plt.xlabel('parameter (n_neighbors)')
# plt.ylabel('accuracy')
# plt.title('Accuracy grid search KNN with best param:'+ str(best_params))
# plt.show()
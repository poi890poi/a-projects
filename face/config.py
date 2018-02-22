from sklearn.svm import LinearSVC, SVC
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier

class HyperParam():
    #window_size = [96, 64]
    window_size = [72, 48]
    window_stride = [8, 8]
    cell_size = [8, 8]
    block_size = [2, 2]
    nbins = 9

class Model():
    @staticmethod
    def svc():
        #return GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
        #return GridSearchCV(SVC(), {'C': [1.0, 2.0, 4.0, 8.0]})

        # specify parameters and distributions to sample from
        param_dist = {"max_depth": [5, None],
                    "max_features": sp_randint(8, 32),
                    "min_samples_split": sp_randint(2, 16),
                    "min_samples_leaf": sp_randint(1, 16),
                    "bootstrap": [True, False],
                    "criterion": ["gini", "entropy"]}

        clf = RandomForestClassifier(n_estimators=128)

        # run randomized search
        n_iter_search = 128
        return RandomizedSearchCV(clf, param_distributions=param_dist,
                                        n_iter=n_iter_search)
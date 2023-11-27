from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def classifier_importance(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)

    pred = clf.predict(x_test)
    score = f1_score(y_test, pred, average='macro')
    return score


def decision_tree_importance(x_train, y_train, x_test, y_test):
    clf = DecisionTreeClassifier(random_state=0, min_impurity_decrease=0.0001)
    return classifier_importance(clf, x_train, y_train, x_test, y_test)


def randon_forest_importance(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, min_impurity_decrease=0.0001, n_jobs=1)
    return classifier_importance(clf, x_train, y_train, x_test, y_test)


def svm_importance(x_train, y_train, x_test, y_test):
    from sklearn.svm import SVC
    clf = SVC(gamma='auto')
    return classifier_importance(clf, x_train, y_train, x_test, y_test)


def knn_importance(x_train, y_train, x_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    return classifier_importance(clf, x_train, y_train, x_test, y_test)


def logistic_regression_importance(x_train, y_train, x_test, y_test):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0, max_iter=1000)
    return classifier_importance(clf, x_train, y_train, x_test, y_test)


def nn_importance(x_train, y_train, x_test, y_test):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(random_state=1, max_iter=1000)
    return classifier_importance(clf, x_train, y_train, x_test, y_test)


class ParallelFeatureImportance:
    def __init__(self, model: Callable, workers: int = 8):
        self.model = model
        self.workers = workers

    def _run(self, fi: int, x_train, y_train, x_test, y_test):
        x_train_fi = x_train[:, fi, :]
        x_test_fi = x_test[:, fi, :]
        score = self.model(x_train_fi, y_train, x_test_fi, y_test)
        return score

    def __call__(self, x_train, y_train, x_test, y_test):
        scores = {}
        features = list(range(x_train.shape[1]))
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {}
            for fi in features:
                future = executor.submit(self._run, fi, x_train, y_train, x_test, y_test)
                futures[future] = fi

            for future in as_completed(futures):
                fi = futures[future]
                scores[fi] = future.result()

        return [scores[fi] for fi in features]

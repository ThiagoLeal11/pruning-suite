import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

from sklearn.metrics import f1_score


import pruning_suite.common as c
from pruning_suite.common import NAMED_MODULES, NAMED_RATIO, NAMED_IMPORTANCE


class ClassifierBasedImportance(c.GenericImportance):
    def __init__(self, estimator: str = 'decision_tree', n_jobs: int = 0):
        if n_jobs <= 1:
            # Get the number of cores
            n_jobs = multiprocessing.cpu_count() - 2

        clf = None
        if estimator == 'decision_tree':
            clf = decision_tree_importance
        elif estimator == 'random_forest':
            clf = randon_forest_importance
        elif estimator == 'svm':
            clf = svm_importance
        elif estimator == 'knn':
            clf = knn_importance
        elif estimator == 'logistic_regression':
            clf = logistic_regression_importance
        elif estimator == 'nn':
            clf = nn_importance

        if not clf:
            raise Exception(f'Estimator {estimator} not found')

        self.estimator = ParallelFeatureImportance(clf, workers=n_jobs)

    def eval_features(self, model, data: c.PruningDataset, to_prune_modules: NAMED_MODULES, prune_ratio: NAMED_RATIO) -> NAMED_IMPORTANCE:
        # Get the features for training and test
        # TODO: Need to be more clear about the need of extraction features
        print(' > Extracting features')
        x, y = next(iter(data.train))
        train = c.extract_features(model, to_prune_modules, x, y)
        x, y = next(iter(data.test))
        test = c.extract_features(model, to_prune_modules, x, y)

        # Get the pruning ratio for each module
        print(' > Evaluating features')
        modules_feature_importance = {}
        for m in to_prune_modules.keys():
            x_train = c.named_batch_features_flatten(c.hydrate_named_features(train.x[m]))
            x_test = c.named_batch_features_flatten(c.hydrate_named_features(test.x[m]))
            already_pruned = c.get_pruned_features(m)

            named_features_ranking = {}
            for name in x_train.keys():
                ratio = c.get_ratio(prune_ratio, [c.full_class_name(m), name])
                if c.to_low_ratio(ratio):
                    continue

                # skip the already pruned layers
                pruned = already_pruned.get(name, None)
                rank = self.estimator(
                    x_train=c.select_features(x_train[name], pruned),
                    y_train=c.select_features(train.y, pruned),
                    x_test=c.select_features(x_test[name], pruned),
                    y_test=c.select_features(test.y, pruned),
                )
                rank = c.fill_gaps(rank, pruned, 1)
                named_features_ranking[name] = rank
            modules_feature_importance[m] = named_features_ranking

        return modules_feature_importance


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


def classifier_importance(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)

    pred = clf.predict(x_test)
    score = f1_score(y_test, pred, average='macro')
    return score


def decision_tree_importance(x_train, y_train, x_test, y_test):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=0, min_impurity_decrease=0.0001)
    return classifier_importance(clf, x_train, y_train, x_test, y_test)


def randon_forest_importance(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
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

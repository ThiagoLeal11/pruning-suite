import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

import pruning_suite.common as c
from pruning_suite.common import PruningDataset, NAMED_MODULES, NAMED_RATIO, NAMED_IMPORTANCE


def _matrix_rank(arr: torch.tensor, n_features: int):
    return [
        torch.linalg.matrix_rank(arr[j, :]).item()
        for j in range(n_features)
    ]


class HRankImportance(c.GenericImportance):
    def _parallel_matrix_rank(self, x, n_batch, n_features):
        results = []
        feature_results = []
        assert len(x.shape) == 4
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {}
            for i in range(n_batch):
                future = executor.submit(_matrix_rank, x[i, :], n_features)
                futures[future] = i

            for future in as_completed(futures):
                item = futures[future]
                result = future.result()
                feature_results.append((item, result))

            sorted_results = sorted(feature_results, key=lambda arr: arr[0])
            for k in sorted_results:
                results += k[1]

        return results

    def eval_features(self, model, data: PruningDataset, to_prune_modules: NAMED_MODULES,
                      prune_ratio: NAMED_RATIO) -> NAMED_IMPORTANCE:
        # TODO: Need to be more clear about the need of extraction features
        print(' > Extracting features')
        x, y = next(iter(data.train))
        train = c.extract_features(model, to_prune_modules, x, y)

        # Get the pruning ratio for each module
        print(' > Evaluating features')
        modules_feature_importance = {}

        # Accumulate the importance of each feature
        features_result = torch.tensor(0.)
        total = torch.tensor(0.)
        for m in to_prune_modules.keys():
            x_train = c.hydrate_named_features(train.x[m])
            already_pruned = c.get_pruned_features(m)

            named_features_ranking = {}
            for name in x_train.keys():
                n_batch, n_features = x_train[name].shape[:2]

                if not features_result.shape or features_result.shape[0] != n_features:
                    features_result = torch.tensor(0.)
                    total = torch.tensor(0.)

                # TODO: Skip layers with low ratio
                # TODO: skip features already pruned
                local_rank = torch.tensor(
                    self._parallel_matrix_rank(x_train[name], n_batch, n_features)
                ).view(n_batch, -1).float().sum(0)

                features_result = (features_result * total) + local_rank
                total += n_batch
                features_result /= total

                pruned = already_pruned.get(name, None)
                upper_limit = 52e32
                named_features_ranking[name] = c.replace_when(features_result, pruned, upper_limit)
            modules_feature_importance[m] = named_features_ranking

        return modules_feature_importance

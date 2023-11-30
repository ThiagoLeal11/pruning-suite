import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import IO, Optional
from torch.utils.data import DataLoader

import torch

MODULE = any
TENSOR = torch.tensor
NAMED_MODULES = dict[MODULE, str]
NAMED_FEATURES = dict[str, TENSOR]
NAMED_IMPORTANCE = dict[MODULE, NAMED_FEATURES]
NAMED_RATIO = dict[str, float | dict[str, float]]
NAMED_FEATURES_PRUNED = dict[str, list[bool]]


@dataclass
class PruningDataset:
    train: DataLoader
    test: DataLoader


@dataclass
class FeaturesData:
    x: dict[MODULE, IO]
    y: TENSOR


@dataclass
class LabeledData:
    x: TENSOR
    y: TENSOR


class GenericImportance(ABC):
    @abstractmethod
    def eval_features(self, model, data: PruningDataset, to_prune_modules: NAMED_MODULES,
                      prune_ratio: NAMED_RATIO) -> NAMED_IMPORTANCE:
        raise NotImplementedError


def full_class_name(c) -> str:
    return f'{c.__class__.__module__}.{c.__class__.__name__}'


def get_ratio(ratio_value: float | NAMED_RATIO, names: list[str]) -> float:
    if isinstance(ratio_value, float):
        return ratio_value

    if len(names) == 0:
        raise Exception('Missing name for ratio')

    name = names.pop(0)
    return get_ratio(ratio_value.get(name, 0), names)


# TODO: put 'features_output' under a constant name
def extract_features(model, to_prune_modules: NAMED_MODULES, x: TENSOR, y: TENSOR) -> FeaturesData:
    model(x)
    data = FeaturesData(
        x={
            m: m.features_output
            for m in to_prune_modules.keys()
        },
        y=y
    )
    for m in to_prune_modules.keys():
        if hasattr(m, 'features_output'):
            del m.features_output

    return data


def named_batch_features_flatten(x: NAMED_FEATURES) -> NAMED_FEATURES:
    # Flatten all the features keeping only the batch and the features dimensions.
    data = {}
    for name, features in x.items():
        batch, num_features = features.shape[0], features.shape[1]
        d = features.reshape(batch, num_features, -1).detach()
        data[name] = d
    return data


# TODO: put 'features_output' under a constant name
def dehydrate_named_feature(self, x, *args, **kwargs):
    features_output = self.extract_feature_middleware(x, *args, **kwargs)

    temp = tempfile.NamedTemporaryFile(prefix='prune_features_extraction', suffix='.pth')
    torch.save(features_output, temp.name)
    temp.flush()  # Force save to disk
    self.features_output = temp

    return self.original_forward(x, *args, **kwargs)


def hydrate_named_features(x: IO) -> NAMED_FEATURES:
    return torch.load(x.name)


def to_low_ratio(ratio: float, threshold: float = 0.0001) -> bool:
    return ratio <= threshold


def select_features(features: TENSOR, mask_out: Optional[list[bool]]) -> TENSOR:
    if not mask_out:
        return features

    if len(mask_out) != features.shape[1]:
        raise Exception(f'Invalid mask length: {len(mask_out)} != {features.shape[1]}')

    return features[:, mask_out, :]


def fill_gaps(x: list[float], should_fill: Optional[list[bool]], fill_value: int) -> list[float]:
    if not should_fill:
        return x

    result = []
    for s in should_fill:
        if s:
            result.append(fill_value)
        else:
            result.append(x.pop(0))

    return result


def _int_to_bool(x: list[int]) -> list[bool]:
    return [True if i > 0 else False for i in x]


def _bool_to_int(x: list[bool]) -> list[int]:
    return [1 if i else 0 for i in x]


# TODO: Put 'features_pruned' under a constant name
def get_pruned_features(m: MODULE) -> NAMED_FEATURES_PRUNED:
    if hasattr(m, 'features_pruned'):
        return {
            k: _int_to_bool(v)
            for k, v in m.features_pruned.items()
        }

    return {}


# TODO: Put 'features_pruned' under a constant name
def update_pruned_features(m: MODULE, features_pruned: NAMED_FEATURES_PRUNED):
    if not hasattr(m, 'features_pruned'):
        m.features_pruned = {
            k: _bool_to_int(v)
            for k, v in features_pruned.items()
        }
        return get_pruned_features(m)

    for k, new in features_pruned.items():
        old = m.features_pruned[k]
        pruned = [int(i+j) for i, j in zip(new, old)]
        m.features_pruned[k] = pruned

        if any(p for p in pruned if p > 1):
            print(f'WARNING: {k} features pruned more than once')

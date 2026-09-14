import sys
from importlib import import_module
from types import ModuleType

from qemscore.datasets.schema import FEATURE_SPEC_VERSION, FEATURES, build_features
from qemscore.datasets.splits import SplitSpec, resolve_split_spec

_LAZY_EXPORTS = {
    "PRESETS": ("qemscore.datasets.generate", "PRESETS"),
    "generate": ("qemscore.datasets.generate", "generate"),
    "SPLIT_PRESETS": ("qemscore.datasets.split_generate", "SPLIT_PRESETS"),
    "generate_split": ("qemscore.datasets.split_generate", "generate_split"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name)
    for export, (export_module, export_attribute) in _LAZY_EXPORTS.items():
        if export_module == module_name:
            globals()[export] = getattr(module, export_attribute)
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


class _DatasetsModule(ModuleType):
    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        target = _LAZY_EXPORTS.get(name)
        if target is not None and isinstance(value, ModuleType):
            value = getattr(value, target[1])
            super().__setattr__(name, value)
        return value


sys.modules[__name__].__class__ = _DatasetsModule

__all__ = [
    "PRESETS",
    "SPLIT_PRESETS",
    "SplitSpec",
    "generate",
    "generate_split",
    "resolve_split_spec",
    "FEATURES",
    "FEATURE_SPEC_VERSION",
    "build_features",
]

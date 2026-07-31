# -*- coding: utf-8 -*-

import importlib
import warnings


_MODEL_PACKAGES = ("sba", "gru", "lstm")

for _pkg in _MODEL_PACKAGES:
    try:
        importlib.import_module(f"{__name__}.{_pkg}")
    except Exception as exc:
        warnings.warn(f"Failed to import custom model package '{_pkg}': {exc}", RuntimeWarning)

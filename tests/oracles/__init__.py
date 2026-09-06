"""Independently written reference calculations used to check estimators.

Modules here are deliberately slow and plain. They exist so a test can compare a
production estimator against a second calculation that shares none of its code,
so nothing in ``qem_bench`` may import from this package.
"""

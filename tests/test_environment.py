"""Smoke test: core imports resolve (CI runs without notebook data)."""


def test_import_stack():
    import joblib  # noqa: F401
    import matplotlib.pyplot  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import seaborn  # noqa: F401
    import scipy.stats  # noqa: F401
    import sklearn.ensemble  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    import sklearn.metrics  # noqa: F401
    import sklearn.model_selection  # noqa: F401
    import sklearn.preprocessing  # noqa: F401
    import sklearn.svm  # noqa: F401
    import sklearn.tree  # noqa: F401
    import statsmodels.stats.outliers_influence  # noqa: F401
    import xgboost  # noqa: F401

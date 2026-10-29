import pandas as pd
import pytest

from src import config, data


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


def test_load_and_schema_valid(raw):
    assert raw.shape == (1025, 14)
    data.validate_schema(raw)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        data.load_raw("does_not_exist.csv")


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.drop(columns=["age"]), "Column mismatch"),
    (lambda d: d.assign(extra=1), "Column mismatch"),
    (lambda d: d.assign(age=d["age"].astype(float).where(d.index != 0)), "Missing values"),
    (lambda d: d.assign(cp=d["cp"].where(d.index != 0, 9)), "unexpected codes"),
    (lambda d: d.assign(target=d["target"].where(d.index != 0, 2)), "Target"),
    (lambda d: d.assign(chol=d["chol"].where(d.index != 0, 5000)), "outside"),
    (lambda d: d.assign(age=d["age"].astype(str)), "numeric"),
])
def test_schema_violations(raw, mutate, msg):
    with pytest.raises(data.SchemaError, match=msg):
        data.validate_schema(mutate(raw.copy()))


def test_duplicate_summary_and_dedup(raw):
    s = data.duplicate_summary(raw)
    assert s["rows"] == 1025 and s["unique_rows"] == 302 and s["redundant_rows"] == 723
    assert s["unique_rows_with_same_features_but_different_label"] == 0
    u = data.drop_exact_duplicates(raw)
    assert len(u) == 302 and not u.duplicated().any()
    assert u.index.tolist() == list(range(302))


def test_dedup_synthetic():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    assert len(data.drop_exact_duplicates(df)) == 2


def test_feature_target_separation(raw):
    X, y = data.split_features_target(raw)
    assert list(X.columns) == config.FEATURES and config.TARGET not in X.columns
    assert y.name == config.TARGET and len(X) == len(y)


def test_split_is_disjoint_stratified_reproducible(raw):
    u = data.drop_exact_duplicates(raw)
    Xtr, Xte, ytr, yte = data.make_split(u)
    key = lambda f: set(f.astype(str).agg("|".join, axis=1))
    assert not key(Xtr) & key(Xte)
    assert len(Xtr) + len(Xte) == len(u)
    assert abs(ytr.mean() - yte.mean()) < 0.05
    Xtr2, Xte2, *_ = data.make_split(u)
    assert Xtr.index.equals(Xtr2.index) and Xte.index.equals(Xte2.index)

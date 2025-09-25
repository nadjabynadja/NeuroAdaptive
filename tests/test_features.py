import numpy as np

from neuroadaptive.features import extract_features, feature_vector


def test_extract_features_shapes():
    samples = np.random.randn(4, 512)
    feats = extract_features(samples, fs=256)
    vec = feature_vector(feats)
    assert len(feats) == vec.shape[0]
    assert vec.ndim == 1

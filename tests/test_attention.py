import numpy as np

from phase3_transformers.attention_numpy import (causal_mask, init_mha_params, multi_head_attention,
                                                scaled_dot_product_attention, softmax)


def test_softmax_rows_sum_to_one():
    x = np.random.default_rng(0).normal(size=(4, 6)) * 50      # grandes valeurs : teste la stabilité
    assert np.allclose(softmax(x).sum(axis=-1), 1.0)


def test_attention_output_is_weighted_average_of_values():
    rng = np.random.default_rng(1)
    Q, K, V = (rng.normal(size=(5, 4)) for _ in range(3))
    out, w = scaled_dot_product_attention(Q, K, V)
    assert np.allclose(w.sum(axis=-1), 1.0) and np.allclose(out, w @ V)


def test_causal_mask_blocks_future_positions():
    rng = np.random.default_rng(2)
    Q, K, V = (rng.normal(size=(6, 4)) for _ in range(3))
    _, w = scaled_dot_product_attention(Q, K, V, mask=causal_mask(6))
    assert np.allclose(np.triu(w, k=1), 0.0)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_attention_is_permutation_equivariant_without_positions():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5, 8))
    params = init_mha_params(8, 2, seed=0)
    perm = rng.permutation(5)
    out, _ = multi_head_attention(X, params, 2)
    out_perm, _ = multi_head_attention(X[perm], params, 2)
    assert np.allclose(out[perm], out_perm)


def test_one_head_matches_single_attention():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(5, 8))
    params = init_mha_params(8, 1, seed=1)
    single, _ = scaled_dot_product_attention(X @ params["Wq"], X @ params["Wk"], X @ params["Wv"])
    multi, _ = multi_head_attention(X, params, 1)
    assert np.allclose(multi, single @ params["Wo"])


def test_multi_head_shapes():
    X = np.random.default_rng(5).normal(size=(7, 12))
    out, w = multi_head_attention(X, init_mha_params(12, 3), 3)
    assert out.shape == (7, 12) and w.shape == (3, 7, 7)

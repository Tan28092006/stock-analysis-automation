from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _get_quasi_diag(link: np.ndarray) -> list[int]:
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = (df0.values - num_items).astype(int)
        sort_ix.loc[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(cov: pd.DataFrame, items: list[str]) -> float:
    sub_cov = cov.loc[items, items]
    inv_diag = 1.0 / np.diag(sub_cov.values)
    weights = inv_diag / inv_diag.sum()
    return float(weights @ sub_cov.values @ weights)


def hrp_weights(close_prices: pd.DataFrame, max_weight: float = 0.25) -> dict[str, float]:
    if close_prices.shape[1] == 0:
        return {}
    if close_prices.shape[1] == 1:
        return {close_prices.columns[0]: round(float(min(1.0, max_weight)), 4)}

    returns = close_prices.pct_change().dropna(how="all").fillna(0)
    cov = returns.cov()
    corr = returns.corr().clip(-0.999, 0.999).fillna(0)
    dist = np.sqrt((1 - corr) / 2)
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="single")
    sort_ix = _get_quasi_diag(link)
    sorted_symbols = corr.index[sort_ix].tolist()

    weights = pd.Series(1.0, index=sorted_symbols)
    clusters = [sorted_symbols]
    while clusters:
        clusters = [
            cluster[start:end]
            for cluster in clusters
            for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
            if len(cluster) > 1
        ]
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                continue
            left = clusters[i]
            right = clusters[i + 1]
            left_var = _cluster_var(cov, left)
            right_var = _cluster_var(cov, right)
            alpha = 1 - left_var / (left_var + right_var) if (left_var + right_var) else 0.5
            weights[left] *= alpha
            weights[right] *= 1 - alpha

    weights = weights / weights.sum()
    weights = _cap_and_redistribute(weights, max_weight)
    return {symbol: round(float(weight), 4) for symbol, weight in weights.items()}


def _cap_and_redistribute(weights: pd.Series, max_weight: float) -> pd.Series:
    weights = weights.copy().astype(float)
    total = float(weights.sum())
    if total <= 0:
        return weights
    weights = weights / total
    if max_weight <= 0:
        return weights
    for _ in range(20):
        over = weights > max_weight
        if not over.any():
            break
        excess = float((weights[over] - max_weight).sum())
        weights[over] = max_weight
        under = ~over
        if not under.any() or weights[under].sum() <= 0:
            break
        room = (max_weight - weights[under]).clip(lower=0)
        if room.sum() <= 0:
            break
        redistribution = weights[under] / weights[under].sum() * excess
        weights[under] += np.minimum(redistribution, room)
    return weights.clip(upper=max_weight)


def equal_score_weights(scores: dict[str, float], max_weight: float = 0.25) -> dict[str, float]:
    if not scores:
        return {}
    if len(scores) == 1:
        symbol = next(iter(scores))
        return {symbol: round(float(min(1.0, max_weight)), 4)}
    raw = pd.Series({k: max(v, 1.0) for k, v in scores.items()}, dtype=float)
    weights = raw / raw.sum()
    weights = _cap_and_redistribute(weights, max_weight)
    return {symbol: round(float(weight), 4) for symbol, weight in weights.items()}

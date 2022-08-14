import numpy as np

def resample_regular(l, n):
    """Resamples a given list to have length `n`.

    List elements are repeated or removed at regular intervals to reach the desired length.

    :param list l: list to resample
    :param int n: desired length of resampled list
    :return list: the resampled list of length `n`

    :Example:

    >>> resample_regular([0, 1, 2, 3, 4, 5], n=3)
    [0, 2, 4]
    >>> resample_regular([0, 1, 2, 3], n=6)
    [0, 1, 2, 3, 0, 2]
    """
    n = int(n)
    if n <= 0:
        return []

    if len(l) == 0:
        return l

    if len(l) < n:  # List smaller than n (Repeat elements)
        resampling_idxs = list(range(len(l))) * (n // len(l))  # Full repetitions

        if len(resampling_idxs) < n:  # Partial repetitions
            resampling_idxs += np.round(np.arange(
                start=0., stop=float(len(l)) - 1., step=len(l) / float(n % len(l))), decimals=0).astype(int).tolist()

        assert len(resampling_idxs) == n
        return [l[i] for i in resampling_idxs]
    elif len(l) > n:  # List bigger than n (Subsample elements)
        resampling_idxs = np.round(np.arange(
            start=0., stop=float(len(l)) - 1., step=len(l) / float(n)), decimals=0).astype(int).tolist()

        assert len(resampling_idxs) == n
        return [l[i] for i in resampling_idxs]
    else:
        return l
    
    
def split_list(l, fraction=None, indexes=None):
    """Splits a given list in two sub-lists ``(a, b)`` either by fraction or by indexes. Only one of the two options should be different to None.
    :param list l: list to split
    :param float fraction: (default: None) fraction of elements for list a (0.0 < fraction < 1.0).
    :param List[int] indexes: (default: None) list of integer indexes of elements for list ``a``.
    :return: tuple of two lists ``(a, b)`` where ``a`` contains the given fraction or indexes and ``b`` the rest.
    :Example:
    >>> split_list([0, 1, 2, 3, 4, 5], fraction=0.5)
    ([0, 1, 2], [3, 4, 5])
    >>> split_list(['a', 'b', 'c', 'd'], fraction=0.75)
    (['a', 'b', 'c'], ['d'])
    >>> split_list([0, 1, 2, 3, 4, 5], indexes=[0, 2, 4])
    ([0, 2, 4], [1, 3, 5])
    >>> split_list(['a', 'b', 'c', 'd'], indexes=[0, 3])
    (['a', 'd'], ['b', 'c'])
    """
    assert any([fraction is None, indexes is None]) and not all([fraction is None, indexes is None])

    if fraction is not None:
        assert isinstance(l, list) and 0.0 < fraction < 1.0
        split_idx = int(np.ceil(len(l) * fraction))
        return l[:split_idx], l[split_idx:]
    else:  # indexes is not None
        assert isinstance(l, list) and all([isinstance(idx, int) for idx in indexes])
        list_a = [a for n, a in enumerate(l) if n in indexes]
        list_b = [b for n, b in enumerate(l) if n not in indexes]
        return list_a, list_b
    
    
def split_dict(d, fraction=None, indexes=None):
    d1_keys, d2_keys = split_list(list(d.keys()), fraction=fraction, indexes=indexes)
    return {k: d[k] for k in d1_keys}, {k: d[k] for k in d2_keys}

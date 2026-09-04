import warnings

import pandas as pd


def find_one_cluster(
    df,
    columns,
    min_cluster_size=None,
    min_samples=None,
    n_cluster=2,
    eps=None,
    allow_single_cluster=True,
    mode="HDBSCAN",
    cluster_label_column="cluster_labels",
):
    """Run an automated clustering algorithm on selected columns and add the labels to the DataFrame.

    Extracts the given columns and applies HDBSCAN, DBSCAN or a Gaussian Mixture Model
    depending on ``mode``. The resulting labels are written to a new column named
    ``f"{cluster_label_column}_{mode}"`` (for example ``cluster_labels_HDBSCAN``). Points that
    belong to no cluster get the label ``-1``: HDBSCAN and DBSCAN produce those themselves,
    while the Gaussian Mixture Model assigns every usable point to a component.

    Only the columns listed in ``columns`` are used, so the same data can be clustered several
    times in different quantities — the distance on its own, the positions X, Y, Z, or the
    velocities U, V, W — and the results compared. Give each run its own
    ``cluster_label_column`` when doing this, otherwise the later run overwrites the earlier
    one; a warning is raised if a label column is about to be replaced.

    Rows with a missing value in any of the feature columns cannot be clustered by any of the
    three algorithms, so they are left out of the fit, labelled ``-1`` and reported in a
    warning. This happens routinely after convert_to_galactic, which writes NaN for stars
    whose parallax could not be used.

    Args:
        df (pandas.DataFrame): Input DataFrame. It is modified in place.
        columns (list of str): Column names to use as clustering features.
        min_cluster_size (int, optional): Smallest group of stars HDBSCAN is allowed to call
            a cluster. Used by HDBSCAN only. Left at None, it is chosen from the amount of data
            being clustered: one percent of the stars, but never fewer than 20. A small fixed
            value does not survive the jump from a test table to a real catalogue -- with a few
            thousand field stars and no empty space between them, HDBSCAN will happily call the
            whole field one cluster. The chosen value is printed. Pass your own number to see
            how much the result depends on it. Defaults to None.
        min_samples (int, optional): How many neighbours a star needs within ``eps`` for DBSCAN
            to treat it as part of a dense region. Used by DBSCAN only. Left at None, it is
            chosen the same way as min_cluster_size: one percent of the stars being clustered,
            never fewer than 20. Defaults to None.
        n_cluster (int): Number of components for the Gaussian Mixture Model. Used by the
            Gaussian Mixture Model only. Defaults to 2.
        eps (float, optional): Radius of the neighbourhood DBSCAN searches around each star, in
            the units of the feature columns. No single value means the same thing in every
            feature space -- half a degree on the sky and half a parsec in X, Y, Z are worlds
            apart -- so left at None it is measured from the data: the distance to the
            min_samples-th nearest neighbour, taken at the tenth percentile, which is small
            enough that only the denser tenth of the field can start a cluster. The chosen
            value is printed. Used by DBSCAN only. Defaults to None.
        allow_single_cluster (bool): Whether HDBSCAN may report a single cluster. This has to
            stay True for the usual exercise of finding one stellar cluster among field stars;
            with False, HDBSCAN refuses to return the one cluster and labels everything as
            noise. Used by HDBSCAN only. Defaults to True.
        mode (str): Which algorithm to use: "HDBSCAN", "DBSCAN" or "GMM". Defaults to "HDBSCAN".
        cluster_label_column (str): Base name of the new label column; the mode is appended.
            Defaults to "cluster_labels".

    Returns:
        pandas.DataFrame: The same DataFrame with the new label column added.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")
    if not isinstance(columns, list):
        raise TypeError("Input columns must be a list.")
    if not all(isinstance(col, str) for col in columns):
        raise TypeError("All elements in columns list must be strings.")
    if min_cluster_size is not None and not isinstance(min_cluster_size, int):
        raise TypeError("min_cluster_size must be an integer or None.")
    if isinstance(min_cluster_size, int) and min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2.")
    if min_samples is not None and not isinstance(min_samples, int):
        raise TypeError("min_samples must be an integer or None.")
    if isinstance(min_samples, int) and min_samples < 1:
        raise ValueError("min_samples must be at least 1.")
    if not isinstance(n_cluster, int):
        raise TypeError("n_cluster must be an integer.")
    if eps is not None and not isinstance(eps, (int, float)):
        raise TypeError("eps must be a number or None.")
    if eps is not None and eps <= 0:
        raise ValueError("eps must be larger than zero.")
    if not isinstance(allow_single_cluster, bool):
        raise TypeError("allow_single_cluster must be True or False.")
    if not isinstance(mode, str):
        raise TypeError("mode must be a string.")
    if not isinstance(cluster_label_column, str):
        raise TypeError("cluster_label_column must be a string.")
    if mode not in ("HDBSCAN", "DBSCAN", "GMM"):
        raise ValueError("Mode must be either 'HDBSCAN', 'DBSCAN' or 'GMM'.")

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"These columns are not in the DataFrame: {missing}. Available columns: {list(df.columns)}")

    # None of the algorithms can handle missing values, so cluster only the complete rows and
    # mark the rest as noise (-1). Stars dropped by convert_to_galactic end up here.
    data = df[columns]
    complete = data.notna().all(axis=1)
    n_incomplete = int((~complete).sum())
    if n_incomplete:
        warnings.warn(
            f"{n_incomplete} of {len(df)} rows have a missing value in {columns}; "
            f"they are excluded from the clustering and labelled -1.",
            stacklevel=2,
        )
    if not complete.any():
        raise ValueError(f"No row has a value in every one of the columns {columns}, so there is nothing to cluster.")

    # How small a group still counts as a cluster only means something next to the amount of
    # data. One percent of the stars keeps the same call working on a 300-row test table and on
    # a 15000-row Gaia field, where a fixed small value would return the whole field as one
    # cluster.
    n_usable = int(complete.sum())
    if mode == "HDBSCAN":
        if min_cluster_size is None:
            min_cluster_size = max(20, round(n_usable / 100))
        print(f"find_one_cluster: using min_cluster_size={min_cluster_size} for {n_usable} stars.")

        from sklearn.cluster import HDBSCAN

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            allow_single_cluster=allow_single_cluster,
            copy=True,
        )
    elif mode == "DBSCAN":
        import numpy as np
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors

        if min_samples is None:
            min_samples = max(20, round(n_usable / 100))
            print(f"find_one_cluster: using min_samples={min_samples} for {n_usable} stars.")
        if eps is None:
            # Measure how far apart the stars actually are in these columns instead of guessing
            # a length: for every star, the distance to its min_samples-th nearest neighbour.
            # The tenth percentile of those distances lets only the denser tenth of the field
            # start a cluster, which is where a star cluster sits.
            neighbours = NearestNeighbors(n_neighbors=min_samples + 1).fit(data[complete])
            spacing = neighbours.kneighbors(data[complete])[0][:, -1]
            eps = float(np.percentile(spacing, 10))
            print(f"find_one_cluster: using eps={eps:.3g} in the units of {columns}.")

        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        from sklearn.mixture import GaussianMixture

        clusterer = GaussianMixture(n_components=n_cluster)

    # Comparing several feature spaces means calling this repeatedly, so say something when an
    # earlier result is about to be thrown away instead of losing it silently.
    label_col = cluster_label_column + "_" + mode
    if label_col in df.columns:
        warnings.warn(
            f"Column '{label_col}' already exists and is being overwritten. Pass a different "
            f"cluster_label_column to keep both results side by side.",
            stacklevel=2,
        )

    df[label_col] = -1
    df.loc[complete, label_col] = clusterer.fit_predict(data[complete])

    # Both algorithms that can say "no cluster here" instead answer with one enormous group
    # when the stars do not separate in these columns at all. That looks like a result in the
    # label column and is only obvious next to the total. A Gaussian
    # Mixture Model is left out: it splits every star between its components by construction,
    # so a large component is what it is meant to produce.
    found = df.loc[complete, label_col]
    sizes = found[found >= 0].value_counts()
    if not sizes.empty:
        biggest = int(sizes.iloc[0])
        # Where to draw the line depends on whether anything else was found. A lone group
        # covering more than a third of the stars is the field and nothing else. When other
        # groups exist the large one is the field standing next to them, which is worth saying
        # but only once it is genuinely dominant -- on a real proper-motion run the field can
        # be half the stars while the cluster sits beside it as a perfectly good second group.
        alone = len(sizes) == 1
        share = 0.35 if alone else 0.5
        if mode in ("HDBSCAN", "DBSCAN") and biggest > share * n_usable:
            rest = (
                f"It is the only group, so nothing separated out in {columns} at all. Compare the result "
                f"with the same plot before clustering, and try other columns."
                if alone
                else f"The other {len(sizes) - 1} group(s) can still be real, so look at those rather than "
                f"at the largest one."
            )
            warnings.warn(
                f"The largest group holds {biggest} of the {n_usable} clustered stars "
                f"({biggest / n_usable:.0%}). A group that size is the field itself rather than a star "
                f"cluster. {rest}",
                stacklevel=2,
            )
        if mode == "HDBSCAN" and biggest == min_cluster_size:
            warnings.warn(
                f"The largest group holds exactly min_cluster_size={min_cluster_size} stars. "
                f"That is HDBSCAN cutting the tip off a single peak, so the number describes "
                f"the setting rather than the data. Run it again with a different "
                f"min_cluster_size and see whether the group changes size with it.",
                stacklevel=2,
            )
    return df


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_blobs

    # Three well-separated blobs in three dimensions.
    features = ["X", "Y", "Z"]
    X, _ = make_blobs(n_samples=300, n_features=3, centers=3, cluster_std=0.6, random_state=42)
    df = pd.DataFrame(X, columns=features)

    # Mimic the output of convert_to_galactic, which leaves NaN for unusable parallaxes.
    df.loc[0, "X"] = np.nan

    for mode in ("HDBSCAN", "DBSCAN", "GMM"):
        out = find_one_cluster(df, columns=features, mode=mode, eps=1.0, n_cluster=3)
        label_col = f"cluster_labels_{mode}"

        # Every row keeps a label, the incomplete row is noise, and the blobs are recovered.
        assert label_col in out.columns, f"{mode} did not add its label column"
        assert len(out) == 300, f"{mode} changed the number of rows"
        assert out[label_col].notna().all(), f"{mode} left rows without a label"
        assert out.loc[0, label_col] == -1, f"{mode} should label the NaN row as noise"
        n_clusters = out.loc[out[label_col] >= 0, label_col].nunique()
        assert n_clusters == 3, f"{mode} found {n_clusters} clusters instead of 3"

        print(f"clustering.py: {mode} recovered {n_clusters} clusters and marked the NaN row as noise")

    # The exercise the students actually do: one tight cluster hidden among scattered field
    # stars. HDBSCAN only reports it when allow_single_cluster is True, which is the default.
    rng = np.random.default_rng(42)
    group = rng.normal(0.0, 0.3, size=(60, 3))
    field = rng.uniform(-30.0, 30.0, size=(140, 3))
    one_cluster = pd.DataFrame(np.vstack([group, field]), columns=features)

    found = find_one_cluster(one_cluster, columns=features, min_cluster_size=20)
    labels = found["cluster_labels_HDBSCAN"]
    assert (labels >= 0).any(), "the single cluster was not found; check allow_single_cluster"
    assert (labels[:60] >= 0).sum() >= 20, "too few of the real cluster members were recovered"
    assert (labels[60:] >= 0).sum() == 0, "field stars were wrongly put into the cluster"

    print(f"clustering.py: found the single hidden cluster with {(labels >= 0).sum()} of its 60 members")

    # Students cluster the same stars several times in different quantities and compare the
    # results, so runs with their own label column must sit side by side without clobbering
    # each other -- and a run that would overwrite one has to say so.
    compared = one_cluster.copy()
    for column, name in ((["X"], "distance"), (features, "position")):
        find_one_cluster(compared, columns=column, min_cluster_size=20, cluster_label_column=name)
    assert {"distance_HDBSCAN", "position_HDBSCAN"}.issubset(compared.columns), "runs overwrote each other"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        find_one_cluster(compared, columns=features, min_cluster_size=20, cluster_label_column="position")
    assert any("overwritten" in str(entry.message) for entry in caught), "overwriting a label column did not warn"

    print("clustering.py: separate runs keep their own label columns and warn before overwriting")

    # DBSCAN's eps is a length in the units of the feature columns, so no fixed number can suit
    # both degrees on the sky and parsec in space. Left to itself it has to measure one.
    auto_eps = pd.DataFrame(X, columns=features)
    find_one_cluster(auto_eps, columns=features, mode="DBSCAN", cluster_label_column="auto")
    auto_labels = auto_eps["auto_DBSCAN"]
    assert auto_labels.loc[auto_labels >= 0].nunique() == 3, "DBSCAN did not recover the blobs with a measured eps"

    print("clustering.py: DBSCAN measures its own eps from the data and still finds the blobs")

    # Two ways a run answers confidently with nothing. Neither shows up in the label column, so
    # both have to be said out loud: stars that do not separate at all come back as one group
    # covering the field, and a single smooth peak comes back cut off at min_cluster_size.
    field = pd.DataFrame(rng.uniform(-30.0, 30.0, size=(1000, 3)), columns=features)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        find_one_cluster(field, columns=features, mode="DBSCAN", eps=8.0, min_samples=5, cluster_label_column="all")
    assert any("the field itself" in str(entry.message) for entry in caught), (
        "a group covering the whole field did not warn"
    )

    peak = pd.DataFrame(rng.normal(0.0, 1.0, size=(1000, 3)), columns=features)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        find_one_cluster(peak, columns=features, min_cluster_size=150, cluster_label_column="tip")
    labels = peak["tip_HDBSCAN"]
    assert int(labels.loc[labels >= 0].value_counts().iloc[0]) == 150, "expected the peak to be cut at min_cluster_size"
    assert any("cutting the tip off" in str(entry.message) for entry in caught), (
        "a group cut off at min_cluster_size did not warn"
    )

    print("clustering.py: warns when a group covers the field and when one is cut at min_cluster_size")

import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture


def find_one_cluster(df, columns, min_cluster_size=5, n_cluster=2, mode="HDBSCAN", cluster_label_column="cluster_labels"):
    """
    This function takes a dataframe and a list of column names as input.
    It extracts these columns and then applies either HDBSCAN, DBSCAN or Gaussian Mixture Model to find one cluster in the data set depending on the mode variable.
    It adds the resulting cluster labels to the dataframe under a new column and returns the modified dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list): List of column names to be used for clustering.
        min_cluster_size (int, optional): Minimum cluster size for HDBSCAN and minimum samples for DBSCAN. Not used for Gaussian Mixture Model. Defaults to 5.
        n_cluster (int, optional): Number of components in the Gaussian Mixture Model. Defaults to 2.
        mode (str, optional): Mode of clustering, either 'HDBSCAN', 'DBSCAN' or 'GMM'. Defaults to "HDBSCAN".
        cluster_label_column (str, optional): Name of the new column to store the cluster labels. Defaults to "cluster_labels + {mode}".

    Returns:
        pd.DataFrame: Modified dataframe with the cluster labels added.
    """
    # Check input types
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")
    if not isinstance(columns, list):
        raise TypeError("Input columns must be a list.")
    if not all(isinstance(col, str) for col in columns):
        raise TypeError("All elements in columns list must be strings.")
    if not isinstance(min_cluster_size, int):
        raise TypeError("min_cluster_size must be an integer.")
    if not isinstance(n_cluster, int):
        raise TypeError("n_cluster must be an integer.")
    if not isinstance(mode, str):
        raise TypeError("mode must be a string.")
    if not isinstance(cluster_label_column, str):
        raise TypeError("cluster_label_column must be a string.")

    # Extract columns from dataframe
    data = df[columns]

    # Apply either HDBSCAN, DBSCAN or Gaussian Mixture Model to find one cluster
    if mode == "HDBSCAN":
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    elif mode == "DBSCAN":
        clusterer = DBSCAN(min_samples=min_cluster_size)
    elif mode == "GMM":
        clusterer = GaussianMixture(n_components=n_cluster)
    else:
        raise ValueError("Mode must be either 'HDBSCAN', 'DBSCAN' or 'GMM'.")
    
    cluster_labels = clusterer.fit_predict(data)

    # Add cluster labels to the dataframe
    col_name = cluster_label_column + "_" + mode
    df[col_name] = cluster_labels

    # Return results
    return df


if __name__ == "__main__":
    # Generate random test data
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=2, n_clusters_per_class=1, random_state=42)
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])

    # Apply all types of clustering
    df_hdb = find_one_cluster(df, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'], min_cluster_size=5, mode="HDBSCAN")
    df_dbs = find_one_cluster(df, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'], min_cluster_size=5, mode="DBSCAN")
    df_gmm = find_one_cluster(df, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'], min_cluster_size=5, mode="GMM")

    # Print results
    print(df_hdb.head())
    print(df_dbs.head())
    print(df_gmm.head())
    print(df_dbs["cluster_labels_GMM"].unique())

import skfuzzy as fuzz
from skfuzzy import cmeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

scaler = MinMaxScaler()
pca = PCA(n_components=3)

m = 2
n_clusters = 3
epsilon = 0.0001
max_iter = 100

centers = []

cluster_labels = ['Sangat Laris', 'Laris', 'Tidak Laris']

df_pca = pd.read_csv("static/data.csv")


def preprocess_df(df1, df2, df3):
    merged_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
    merged_df.columns = [col.lower() for col in merged_df.columns]
    df_value = merged_df[['stokawal', 'stokakhir',
                          'kuantitasbarang', 'jenisbarang']]
    df_value['jenisbarang'] = pd.factorize(df_value['jenisbarang'])[0]
    df_value.drop_duplicates(inplace=True)
    df_value.to_csv("static/data.csv", index=False)
    df_value_path = "static/data.csv"
    return df_value_path


def initialize_centers(data, n_clusters):
    idx = np.random.randint(0, data.shape[0], n_clusters)
    centers = data[idx]
    return centers


def calculate_membership(data, centers, m, distances=None):
    if distances is None:
        distances = np.linalg.norm(data[:, None, :] - centers, axis=2)

    membership_matrix = 1 / (distances ** (2 / (m - 1)) + 1e-8)
    membership_matrix = membership_matrix / \
        np.sum(membership_matrix, axis=1)[:, None]
    return membership_matrix


def calculate_centers(data, membership_matrix, m):
    numerator = np.dot(membership_matrix.T, data)
    denominator = np.sum(membership_matrix.T, axis=1)[:, None]
    centers = numerator / (denominator + 1e-8)
    return centers


def fuzzy_c_means(data, n_clusters, m, epsilon=0.001, max_iter=10):
    centers = initialize_centers(data, n_clusters)
    distances = None

    for iter in range(max_iter):
        prev_centers = centers.copy()
        membership_matrix = calculate_membership(
            data, centers, m, distances=distances)
        centers = calculate_centers(data, membership_matrix ** m, m)
        distances = np.linalg.norm(data[:, None, :] - centers, axis=2)

        if np.linalg.norm(centers - prev_centers) < epsilon:
            break
    # Calculate FPC
    fpc = np.sum(membership_matrix ** 2) / (data.shape[0] * n_clusters)
    return centers, membership_matrix


centers, membership_matrix = fuzzy_c_means(
    df_pca, n_clusters, m, epsilon=epsilon, max_iter=max_iter)
cluster_membership = np.argmax(membership_matrix, axis=1)


def visualization():
    df_pca['cluster'] = [cluster_labels[c] for c in cluster_membership]
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(preprocess_df.df_pca[:, 0], preprocess_df.df_pca[:, 1],
                         c=cluster_membership, cmap='rainbow')
    legend = ax.legend(*scatter.legend_elements(),
                       loc="upper right", title="Clusters")
    legend.set_title("Clusters")
    for i, label in enumerate(cluster_labels):
        legend.get_texts()[i].set_text(label)
    for i in range(n_clusters):
        plt.scatter(centers[i][0], centers[i][1], marker='x',
                    s=100, linewidhts=2, color='black')

    # add axis labels and title
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.colorbar(scatter)
    plt.savefig("static/test.png")
    visualization_path = "static/test.png"
    return visualization_path

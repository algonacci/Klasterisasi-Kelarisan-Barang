from flask import Flask, render_template, request
import pandas as pd
import module as md
import numpy as np
from sklearn.decomposition import PCA

app = Flask(__name__)
df_processed = pd.read_csv("static/data.csv")
pca = PCA(n_components=3)
df_pca = pca.fit_transform(df_processed)

m = 2
n_clusters = 3
epsilon = 0.0001
max_iter = 100


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/in_scope")
def in_scope():
    return render_template("in_scope.html")


@app.route("/cluster_non_scope", methods=["POST"])
def cluster_non_scope():
    df1 = pd.read_excel(request.files["file1"])
    df2 = pd.read_excel(request.files["file2"])
    df3 = pd.read_excel(request.files["file3"])
    data_path = md.preprocess_df(df1, df2, df3)

    df_pca = pca.fit_transform(pd.read_csv(data_path))
    centers, membership_matrix = md.fuzzy_c_means(
        df_pca, n_clusters, m, epsilon=epsilon, max_iter=max_iter)
    cluster_membership = np.argmax(membership_matrix, axis=1)
    viz = md.visualization(cluster_membership, centers)
    cluster_counts = pd.Series(
        cluster_membership).value_counts().sort_index().to_dict()
    return render_template("cluster_non_scope.html", viz=viz, cluster_counts=cluster_counts)


@app.route("/cluster_in_scope", methods=["POST"])
def cluster_in_scope():
    df1 = pd.read_excel(request.files["file1"])
    df2 = pd.read_excel(request.files["file2"])
    df3 = pd.read_excel(request.files["file3"])
    data_path = md.preprocess_df(df1, df2, df3)

    df_pca = pca.fit_transform(pd.read_csv(data_path))
    centers, membership_matrix = md.fuzzy_c_means(
        df_pca, n_clusters, m, epsilon=epsilon, max_iter=max_iter)
    cluster_membership = np.argmax(membership_matrix, axis=1)
    viz = md.visualization(cluster_membership, centers)
    cluster_counts = pd.Series(
        cluster_membership).value_counts().sort_index().to_dict()
    return render_template("cluster_in_scope.html", viz=viz, cluster_counts=cluster_counts)


if __name__ == "__main__":
    app.run()

# ==================================================
# UNSUPERVISED LEARNING - KMEANS CLUSTERING
# ==================================================

cluster_features = [col for col in numeric_features if col != "virality_score"]

if cluster_features:

    print("\nStarting Unsupervised Learning (K-Means Clustering)...")

    X_clust = df_train_std[cluster_features]

    # -------------------------------
    # Elbow Method
    # -------------------------------

    ks = list(range(2, 11))
    inertias = []

    for k in ks:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_clust)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.grid(True)

    elbow_file = DATA_DIR / "kmeans_elbow.png"
    plt.savefig(elbow_file)
    plt.show()

    # -------------------------------
    # Choose Best K using Silhouette
    # -------------------------------

    best_k = 2
    best_score = -1

    for k in range(2, 11):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_clust)

        score = silhouette_score(X_clust, labels)

        print(f"K={k} | Silhouette Score={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nBest K selected: {best_k}")
    print(f"Best Silhouette Score: {best_score:.4f}")

    # -------------------------------
    # Final KMeans Model
    # -------------------------------

    final_kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )

    final_kmeans.fit(X_clust)

    df_train_std["cluster"] = final_kmeans.labels_

    df_test_std["cluster"] = final_kmeans.predict(
        df_test_std[cluster_features]
    )

    # -------------------------------
    # Cluster Distribution
    # -------------------------------

    print("\nCluster Counts:")
    print(df_train_std["cluster"].value_counts())

    # -------------------------------
    # Cluster Summary
    # -------------------------------

    cluster_summary = (
        df_train_std
        .groupby("cluster")
        .mean(numeric_only=True)
    )

    print("\nCluster Summary:")
    print(cluster_summary)

    cluster_summary.to_csv(
        DATA_DIR / "cluster_summary.csv"
    )

    # -------------------------------
    # Cluster Visualization
    # -------------------------------

    if len(cluster_features) >= 2:

        plt.figure(figsize=(8,6))

        scatter = plt.scatter(
            X_clust.iloc[:,0],
            X_clust.iloc[:,1],
            c=df_train_std["cluster"],
            cmap="viridis",
            alpha=0.7
        )

        plt.xlabel(cluster_features[0])
        plt.ylabel(cluster_features[1])
        plt.title("K-Means Cluster Visualization")

        plt.colorbar(scatter)

        plt.tight_layout()

        cluster_plot = (
            DATA_DIR /
            "cluster_visualization.png"
        )

        plt.savefig(cluster_plot)
        plt.show()

    print("\nUnsupervised Learning Complete.")

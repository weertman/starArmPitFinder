import os
import warnings
import numpy as np
import umap
from sklearn.cluster import MiniBatchKMeans, KMeans
import leidenalg as la
import igraph as ig
from tqdm import tqdm
from embed_boxes import run_vit_embedding

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class ClusteringPipeline:
    def __init__(self, final_embedding_dim=256, mbk_n_clusters=2000,
                 mbk_batch_size=256, umap_n_neighbors=15,
                 umap_min_dist=0.1, umap_n_components=2,
                 final_kmeans_n_clusters=10, n_samples_per_cluster=2,
                 resolution_parameter=1.0, random_state=42):
        self.final_embedding_dim = final_embedding_dim
        self.mbk_n_clusters = mbk_n_clusters
        self.mbk_batch_size = mbk_batch_size
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_n_components = umap_n_components
        self.final_kmeans_n_clusters = final_kmeans_n_clusters - 1
        self.n_samples_per_cluster = n_samples_per_cluster
        self.resolution_parameter = resolution_parameter
        self.random_state = random_state

    def mini_batch_kmeans(self, embeddings, n_clusters, batch_size):
        if len(embeddings) < n_clusters:
            return embeddings
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.random_state, batch_size=batch_size,
                                 n_init=30)
        kmeans.fit(embeddings)
        return kmeans.cluster_centers_

    def reduce_dimensions(self, embeddings, n_neighbors, min_dist, n_components):
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
                            random_state=self.random_state)
        return reducer.fit_transform(embeddings)

    def construct_graph(self, reduced_embeddings, n_neighbors):
        knn_indices = np.array(
            [np.argsort(np.linalg.norm(reduced_embeddings - p, axis=1))[1:n_neighbors + 1] for p in reduced_embeddings])
        sources = np.repeat(np.arange(reduced_embeddings.shape[0]), n_neighbors)
        targets = knn_indices.flatten()
        weights = np.ones(len(sources))
        g = ig.Graph(edges=list(zip(sources, targets)), edge_attrs={'weight': weights}, directed=False)
        return g

    def leiden_louvain_clustering(self, graph, resolution_parameter):
        partition = la.find_partition(graph, la.RBConfigurationVertexPartition,
                                      resolution_parameter=resolution_parameter)
        labels = np.array(partition.membership)
        print(f"Number leiden of clusters: {len(np.unique(labels))}")
        return labels

    def assign_final_clusters(self, embeddings, leiden_labels, target_n_clusters, random_state):
        final_labels = np.zeros(len(embeddings), dtype=int)
        unique_labels = np.unique(leiden_labels)
        current_n_clusters = len(unique_labels)

        # Calculate additional clusters needed
        additional_clusters_needed = max(0, target_n_clusters - current_n_clusters)

        if additional_clusters_needed == 0:
            return leiden_labels

        # Proportionally distribute additional clusters
        label_counts = np.array([np.sum(leiden_labels == label) for label in unique_labels])
        additional_clusters_per_label = np.round(
            (label_counts / label_counts.sum()) * additional_clusters_needed).astype(int)

        # Ensure total additional clusters match
        while additional_clusters_per_label.sum() > additional_clusters_needed:
            for i in range(len(additional_clusters_per_label)):
                if additional_clusters_per_label[i] > 0:
                    additional_clusters_per_label[i] -= 1
                    if additional_clusters_per_label.sum() == additional_clusters_needed:
                        break

        for i, label in enumerate(unique_labels):
            indices = np.where(leiden_labels == label)[0]
            n_clusters = 1 + additional_clusters_per_label[i]

            if len(indices) <= n_clusters:
                final_labels[indices] = np.arange(len(indices))
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=30)
            kmeans.fit(embeddings[indices])
            final_labels[indices] = kmeans.labels_ + np.max(final_labels) + 1

        return final_labels

    def diversity_sampling(self, labels, n_samples_per_cluster):
        sampled_indices = []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            n_samples = min(n_samples_per_cluster, len(indices))
            sampled_indices.extend(np.random.choice(indices, n_samples, replace=False))
        print(f"Sampled indices: {sampled_indices}")
        return sampled_indices

    def run_clustering_pipeline(self, final_embeddings,
                                mbk_n_clusters,
                                mbk_batch_size,
                                umap_n_neighbors,
                                umap_min_dist,
                                umap_n_components,
                                final_kmeans_n_clusters,
                                n_samples_per_cluster,
                                resolution_parameter,
                                random_state):

        # Step 1: Mini-batch KMeans pre-clustering
        print("Running Mini-batch KMeans pre-clustering...")
        centroids = self.mini_batch_kmeans(final_embeddings, mbk_n_clusters, mbk_batch_size)

        # Step 2: UMAP dimensionality reduction
        print("Running UMAP dimensionality reduction...")
        reduced_centroids = self.reduce_dimensions(centroids, umap_n_neighbors, umap_min_dist, umap_n_components)

        # Step 3: Construct graph from reduced data
        print("Constructing graph...")
        graph = self.construct_graph(reduced_centroids, umap_n_neighbors)

        # Step 4: Leiden-Louvain clustering
        print("Running Leiden-Louvain clustering...")
        leiden_labels = self.leiden_louvain_clustering(graph, resolution_parameter)

        # Step 5: KMeans leiden cluster embeddings
        print("For each leiden cluster, running KMeans clustering...")
        final_labels = self.assign_final_clusters(final_embeddings, leiden_labels, final_kmeans_n_clusters,
                                                  random_state)
        print('Found clusters:', np.unique(final_labels), 'Number of clusters:', len(np.unique(final_labels)))

        # Step 6: Diversity sampling
        print("Sampling images for diversity...")
        sampled_indices = self.diversity_sampling(final_labels, n_samples_per_cluster)
        sampled_indices_clusters = final_labels[sampled_indices]

        return sampled_indices, sampled_indices_clusters

    def run(self, path_root_data):
        print("Clustering pipeline parameters:")
        print(f'Final embedding dimension: {self.final_embedding_dim}')
        print(f'Mini-batch KMeans n_clusters: {self.mbk_n_clusters}')
        print(f'Mini-batch KMeans batch_size: {self.mbk_batch_size}')
        print(f'UMAP n_neighbors: {self.umap_n_neighbors}')
        print(f'UMAP min_dist: {self.umap_min_dist}')
        print(f'UMAP n_components: {self.umap_n_components}')
        print(f'Final KMeans n_clusters: {self.final_kmeans_n_clusters}')
        print(f'Number of samples per cluster: {self.n_samples_per_cluster}')
        print(f'Resolution parameter: {self.resolution_parameter}')
        print(f'Random state: {self.random_state}')
        print()

        # Run ViT embedding
        print("Running ViT embedding...")
        final_embeddings = run_vit_embedding(path_root_data, final_embedding_dim=self.final_embedding_dim,
                                             )

        # Ensure embeddings is a numpy array
        final_embeddings = np.array(final_embeddings)

        # Run the clustering pipeline
        print("Running clustering pipeline...")
        sampled_indices, sampled_indices_clusters = self.run_clustering_pipeline(final_embeddings,
                                                                                   self.mbk_n_clusters,
                                                                                   self.mbk_batch_size,
                                                                                   self.umap_n_neighbors,
                                                                                   self.umap_min_dist,
                                                                                   self.umap_n_components,
                                                                                   self.final_kmeans_n_clusters,
                                                                                   self.n_samples_per_cluster,
                                                                                   self.resolution_parameter,
                                                                                   self.random_state)
        return sampled_indices, sampled_indices_clusters


if __name__ == '__main__':
    path_root_data = os.path.join('..', 'data', 'example')
    cp = ClusteringPipeline()
    sampled_indices = cp.run(path_root_data)

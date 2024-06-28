import os
import pandas as pd
import shutil
from cluster_embeddings import ClusteringPipeline

def open_create_csv(path_root_data):
    path_meta = os.path.join(path_root_data, 'create.csv')
    df = pd.read_csv(path_meta) # pd.DataFrame({'image_path': crop_image_paths, 'label': labels_crop_images})
    return df

def create_cluster_dir (path_root_data):
    clustered_images_dir = os.path.join(path_root_data, 'clustered_images')
    if os.path.exists(clustered_images_dir):
        shutil.rmtree(clustered_images_dir)
    os.makedirs(clustered_images_dir)

    return clustered_images_dir

if __name__ == '__main__':
    path_root_data = os.path.join('..', 'data', 'example')

    # create clustered images dir
    clustered_images_dir = create_cluster_dir(path_root_data)

    # open create.csv
    df = open_create_csv(path_root_data)

    final_embedding_dim = 256
    mbk_n_clusters = 100
    mbk_batch_size = 100
    umap_n_neighbors = 15
    umap_min_dist = 0.1
    umap_n_components = 2
    final_kmeans_n_clusters = 10
    n_samples_per_cluster = 5
    resolution_parameter = 1.0
    random_state = 42

    cp = ClusteringPipeline(final_embedding_dim, mbk_n_clusters,
                            mbk_batch_size, umap_n_neighbors,
                            umap_min_dist, umap_n_components,
                            final_kmeans_n_clusters,
                            n_samples_per_cluster, resolution_parameter,
                            random_state)
    sampled_indices = cp.run(path_root_data)

    # Save sampled indices
    path_sampled_images = df.iloc[sampled_indices]['image_path']
    for i, image_path in enumerate(path_sampled_images):
            shutil.copy(image_path, os.path.join(clustered_images_dir, f'{os.path.basename(image_path)}.jpg'))


import sys
import os

# Add the directory containing `cluster_embeddings.py` to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'training_set_creation')))

from training_set_creation.get_clustered_images import ClusterImages

if __name__ == '__main__':
    path_root_data = os.path.join('..', 'data', 'example')
    if not os.path.exists(path_root_data):
        print(f"Path {path_root_data} does not exist")
        raise FileNotFoundError(f"Path {path_root_data} does not exist")

    clustered_images_dir = ClusterImages(path_root_data,
                                         n_samples_per_cluster=2,
                                         final_kmeans_n_clusters=15,)
    print(f'Clustered images saved to: {clustered_images_dir}')
    print('Done!')
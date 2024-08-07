import os
import pandas as pd
import shutil
import argparse
from cluster_embeddings import ClusteringPipeline
from create_bboxs_from_cvat11 import create_bboxs_from_cvat11

def open_create_csv(path_root_data):
    if not os.path.exists(path_root_data):
        raise FileNotFoundError(f"Path {path_root_data} does not exist")
    else:
        print(f"Found {path_root_data}")

    path_meta = os.path.join(path_root_data, 'create.csv')
    if not os.path.exists(path_meta):
        print('Failed to find create.csv file')
        print(f'Working directory is: {os.getcwd()}')
        print(f"Walking through {path_root_data} to find csv files")
        ## print contents of path_root_data but not subdirs
        for f in os.listdir(path_root_data):
            if f == 'annotations.xml':
                print(f'Found {f}, making boxes')
                _, _ = create_bboxs_from_cvat11(path_root_data)
                df = pd.read_csv(path_meta)  # pd.DataFrame({'image_path': crop_image_paths, 'label': labels_crop_images})
                return df
                break

        raise FileNotFoundError(f"Path {path_meta} does not exist")

    df = pd.read_csv(path_meta) # pd.DataFrame({'image_path': crop_image_paths, 'label': labels_crop_images})
    return df

def create_cluster_dir (path_root_data):
    clustered_images_dir = os.path.join(path_root_data, 'clustered_images')
    if os.path.exists(clustered_images_dir):
        shutil.rmtree(clustered_images_dir)
    os.makedirs(clustered_images_dir)

    return clustered_images_dir

def ClusterImages(path_root_data,
                  final_embedding_dim=256,
                  mbk_n_clusters=2000,
                  mbk_batch_size=500,
                  umap_n_neighbors=25,
                  umap_min_dist=0.1,
                  umap_n_components=15,
                  final_kmeans_n_clusters=200,
                  n_samples_per_cluster=5,
                  resolution_parameter=0.5,
                  random_state=42):

    _,_ = create_bboxs_from_cvat11(path_root_data)

    df = open_create_csv(path_root_data)
    clustered_images_dir = create_cluster_dir(path_root_data)

    cp = ClusteringPipeline(final_embedding_dim, mbk_n_clusters, mbk_batch_size,
                            umap_n_neighbors, umap_min_dist, umap_n_components,
                            final_kmeans_n_clusters, n_samples_per_cluster,
                            resolution_parameter, random_state)
    sampled_indices, sampled_indices_clusters = cp.run(path_root_data)

    sample_labels = df.iloc[sampled_indices]['label']
    unq_labels = sample_labels.unique()
    print(f'Found {len(unq_labels)} unique labels')
    for i, label in enumerate(unq_labels):
        os.mkdir(os.path.join(clustered_images_dir, f'{label}'))

    sample_subdirs = {label:[] for label in unq_labels}
    for label in unq_labels:
        sample_dir = os.path.join(clustered_images_dir, f'{label}')
        sample_subdir = os.path.join(sample_dir, f'{0}')
        os.mkdir(sample_subdir)
        sample_subdirs[label].append(sample_subdir)

    path_sampled_images = df.iloc[sampled_indices]['image_path']
    for i, image_path in enumerate(path_sampled_images):
        cluster = sampled_indices_clusters[i]
        label = sample_labels.iloc[i]
        sample_dir = os.path.join(clustered_images_dir, f'{label}')
        current_subsample_dir = sample_subdirs[label][-1]
        n_images_in_subsample_dir = len(os.listdir(current_subsample_dir))
        if n_images_in_subsample_dir >= 50:
            new_subsample_dir = os.path.join(sample_dir, f'{int(os.path.basename(current_subsample_dir))+1}')
            os.mkdir(new_subsample_dir)
            sample_subdirs[label].append(new_subsample_dir)
            sample_path = os.path.join(new_subsample_dir, f'{cluster}_{os.path.basename(image_path)}.jpg')
        else:
            sample_path = os.path.join(current_subsample_dir, f'{cluster}_{os.path.basename(image_path)}.jpg')

        if os.path.exists(image_path):
            shutil.copy(image_path, sample_path)
        else:
            print(f"File not found: {image_path}")
    return path_sampled_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image clustering pipeline")
    parser.add_argument('--path_root_data',
                        type=str, default=os.path.join( '..', 'data', 'example'),
                        help="Path to the root data directory")
    parser.add_argument('--final_embedding_dim', default=256, type=int, help="Final embedding dimension")
    parser.add_argument('--mbk_n_clusters', default=2000, type=int, help="Mini-batch KMeans n_clusters")
    parser.add_argument('--mbk_batch_size', default=100, type=int, help="Mini-batch KMeans batch_size")
    parser.add_argument('--umap_n_neighbors', default=25, type=int, help="UMAP n_neighbors")
    parser.add_argument('--umap_min_dist', default=0.1, type=float, help="UMAP min_dist")
    parser.add_argument('--umap_n_components', default=15, type=int, help="UMAP n_components")
    parser.add_argument('--final_kmeans_n_clusters', default=200, type=int, help="Final KMeans n_clusters")
    parser.add_argument('--n_samples_per_cluster', default=5, type=int, help="Number of samples per cluster")
    parser.add_argument('--resolution_parameter', default=0.5, type=float, help="Resolution parameter")
    parser.add_argument('--random_state', default=42, type=int, help="Random state")
    args = parser.parse_args()

    path_root_data = args.path_root_data
    clustered_images_dir = create_cluster_dir(path_root_data)
    final_embedding_dim = args.final_embedding_dim
    mbk_n_clusters = args.mbk_n_clusters
    mbk_batch_size = args.mbk_batch_size
    umap_n_neighbors = args.umap_n_neighbors
    umap_min_dist = args.umap_min_dist
    umap_n_components = args.umap_n_components
    final_kmeans_n_clusters = args.final_kmeans_n_clusters
    n_samples_per_cluster = args.n_samples_per_cluster
    resolution_parameter = args.resolution_parameter
    random_state = args.random_state

    path_sampled_images = ClusterImages(path_root_data,
                                        final_embedding_dim,
                                        mbk_n_clusters,
                                        mbk_batch_size,
                                        umap_n_neighbors,
                                        umap_min_dist,
                                        umap_n_components,
                                        final_kmeans_n_clusters,
                                        n_samples_per_cluster,
                                        resolution_parameter,
                                        random_state)
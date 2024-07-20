import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import xml.etree.ElementTree as ET
from multiprocessing import Pool, cpu_count
import glob

def open_root(path_annotations):
    ## open xml file
    tree = ET.parse(path_annotations)
    root = tree.getroot()
    return root

def check_if_label_is_banned(label):
    banned_labels = ["Dead_bivalve", "Dead_sandDollar",
                     "Dead_urchin", "Sand_dollar",
                     "Bivalve", "Unknown_urchin",
                     "Strongylocentrotus droebachiensis",
                     "Strongylocentrotus_franciscanus",
                     "Strongylocentrotus_purpuratus",
                     ]
    if label in banned_labels:
        return True
    return False

def process_image(args):
    child, path_images_dir, path_cropped_images_dir, show, show_till, verbose = args
    image_name = child.attrib['name']
    image_path = os.path.join(path_images_dir, image_name)

    if verbose:
        print('info:', child.tag, child.attrib)
        print(image_path)
        print()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    crop_image_paths = []
    labels_crop_images = []

    for j, polygon in enumerate(child.findall('polygon')):
        label = polygon.attrib['label']

        if check_if_label_is_banned(label):
            continue

        annotation = np.array(
            [[float(coord) for coord in point.split(',')] for point in polygon.attrib['points'].split(';')]
        )

        x_min, y_min = annotation.min(axis=0)
        x_max, y_max = annotation.max(axis=0)
        if verbose:
            print(label)
            print(x_min, y_min, x_max, y_max)

        cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        if show and j < show_till:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(cropped_image)
            plt.show()
            plt.close()

        cropped_image_path = os.path.join(path_cropped_images_dir, f'{label}_{j}_{image_name}')
        cv2.imwrite(cropped_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        crop_image_paths.append(cropped_image_path)
        labels_crop_images.append(label)

    return crop_image_paths, labels_crop_images

def make_bboxes(root, path_images_dir, path_root_data, show=False, show_till=2, verbose=False):
    path_cropped_images_dir = os.path.join(path_root_data, 'cropped_images')
    if not os.path.exists(path_cropped_images_dir):
        os.makedirs(path_cropped_images_dir)

    print('Making bounding boxes from annotations.xml')

    tasks = [(child, path_images_dir, path_cropped_images_dir, show, show_till, verbose) for child in root.findall('image')]

    with Pool(cpu_count()-10) as pool:
        results = list(tqdm(pool.imap(process_image, tasks), total=len(tasks)))

    crop_image_paths = []
    labels_crop_images = []
    for result in results:
        crop_image_paths.extend(result[0])
        labels_crop_images.extend(result[1])

    return crop_image_paths, labels_crop_images

def save_create_meta(crop_image_paths, labels_crop_images, path_root_data):
    path_meta = os.path.join(path_root_data, 'create.csv')
    df = pd.DataFrame({'image_path': crop_image_paths, 'label': labels_crop_images})
    df.to_csv(path_meta, index=False)
    print(f'Created meta file: {path_meta}')

def create_bboxs_from_cvat11(path_root_data):
    path_annotations = os.path.join(path_root_data, 'annotations.xml')
    if not os.path.exists(path_annotations):
        print(f"Path {path_annotations} does not exist")
        print(f'Checking if this the image directory: {path_root_data}')
        print(f'If image directory we will skip making boxs and just get the image paths and labels')
        path_images = glob.glob(os.path.join(path_root_data, '*.jpg'))
        if len(path_images) == 0:
            print(f"Path {path_root_data} does not exist")
            raise FileNotFoundError(f"Path {path_root_data} does not exist")
        else:
            print(f'Found {len(path_images)} images')
            crop_image_paths = path_images
            labels_crop_images = [os.path.basename(path_image).split('_')[0] for path_image in path_images]
            save_create_meta(crop_image_paths, labels_crop_images, path_root_data)
            return crop_image_paths, labels_crop_images

    path_images_dir = os.path.join(path_root_data, 'images')

    root = open_root(path_annotations)
    crop_image_paths, labels_crop_images = make_bboxes(root, path_images_dir, path_root_data, show=False, verbose=False)
    save_create_meta(crop_image_paths, labels_crop_images, path_root_data)

    return crop_image_paths, labels_crop_images

if __name__ == '__main__':
    path_root_data = os.path.join('..', '..', 'data', 'example')
    crop_image_paths, labels_crop_images = create_bboxs_from_cvat11(path_root_data)

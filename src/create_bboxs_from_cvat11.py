import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import xml.etree.ElementTree as ET

def open_root(path_annotations):
    ## open xml file
    tree = ET.parse(path_annotations)
    root = tree.getroot()
    return root

def make_bboxes(root, path_images_dir, path_root_data, show=False, show_till=2, verbose=False):
    path_cropped_images_dir = os.path.join(path_root_data, 'cropped_images')
    if not os.path.exists(path_cropped_images_dir):
        os.makedirs(path_cropped_images_dir)

    ## find all the attributes of the xml file and print them
    crop_image_paths = []
    labels_crop_images = []
    pbar = tqdm(total=len(root.findall('image')), position=0, leave=True)
    for i, child in enumerate(root.findall('image')):
        image_name = child.attrib['name']
        image_path = os.path.join(path_images_dir, image_name)

        if verbose:
            print('info:', child.tag, child.attrib)
            print(image_path)
            print()

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for j, polygon in enumerate(child.findall('polygon')):

            label = polygon.attrib['label']
            annotation = np.array(
                [[float(coord) for coord in point.split(',')] for point in polygon.attrib['points'].split(';')]
            )
            # print(annotation)

            ## min, max box
            x_min, y_min = annotation.min(axis=0)
            x_max, y_max = annotation.max(axis=0)
            if verbose:
                print(label)
                print(x_min, y_min, x_max, y_max)

            ## crop image
            cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            if show and i < show_till:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.imshow(cropped_image)
                plt.show()
                plt.close()

            ## save image
            cropped_image_path = os.path.join(path_cropped_images_dir, f'{label}_{j}_{image_name}')
            cv2.imwrite(cropped_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
            crop_image_paths.append(cropped_image_path)
            labels_crop_images.append(label)

        pbar.update(1)
    pbar.close()

    return crop_image_paths, labels_crop_images

def save_create_meta(crop_image_paths, labels_crop_images, path_root_data):
    path_meta = os.path.join(path_root_data, 'create.csv')
    df = pd.DataFrame({'image_path': crop_image_paths, 'label': labels_crop_images})
    df.to_csv(path_meta, index=False)

    print(f'Create meta file: {path_meta}')

def create_bboxs_from_cvat11(path_root_data):
    path_annotations = os.path.join(path_root_data, 'annotations_dir', 'annotations.xml')
    path_images_dir = os.path.join(path_root_data, 'images')

    root = open_root(path_annotations)
    crop_image_paths, labels_crop_images = make_bboxes(root, path_images_dir, path_root_data, show=False, verbose=False)
    save_create_meta(crop_image_paths, labels_crop_images, path_root_data)

    return crop_image_paths, labels_crop_images

if __name__ == '__main__':
    path_root_data = os.path.join('..', 'data', 'example')
    crop_image_paths, labels_crop_images = create_bboxs_from_cvat11(path_root_data)

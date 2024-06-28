import os
import torch
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from tqdm import tqdm
from create_bboxs_from_cvat11 import create_bboxs_from_cvat11

def load_vit_model(model_name='vit_base_patch16_224'):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    ## check gpu
    if torch.cuda.is_available():
        model.to('cuda')

    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def get_image_embedding(model, image_tensor):
    with torch.no_grad():
        output = model.forward_features(image_tensor)
    return output[0, 0, :].numpy()  # Take the class token embedding

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def embed_images(image_paths, model):
    embeddings = []
    pbar = tqdm(total=len(image_paths), position=0, leave=True)
    for image_path in image_paths:
        image_tensor = preprocess_image(image_path)
        embedding = get_image_embedding(model, image_tensor)
        normalized_embedding = normalize_embedding(embedding)
        embeddings.append(normalized_embedding)
        pbar.update(1)
    pbar.close()
    return embeddings

def normalize_weights(layer):
    with torch.no_grad():
        layer.weight.div_(layer.weight.norm(dim=1, keepdim=True))

def run_vit_embedding(path_root_data, final_embedding_dim=256):
    crop_image_paths, labels_crop_images = create_bboxs_from_cvat11(path_root_data)

    # Load ViT model
    vit_model = load_vit_model()

    # Path to cropped images
    path_cropped_images_dir = os.path.join(path_root_data, 'cropped_images')

    # Get list of cropped image paths
    cropped_image_paths = [os.path.join(path_cropped_images_dir, fname) for fname in os.listdir(path_cropped_images_dir)]

    # Embed images
    embeddings = embed_images(cropped_image_paths, vit_model)

    # Get the original embedding dimension
    original_embedding_dim = embeddings[0].shape[0]
    print(f'Original embedding dimension: {original_embedding_dim}')

    # Add a final linear layer to reduce dimensions to final_embedding_dim
    final_layer = torch.nn.Linear(original_embedding_dim, final_embedding_dim)

    # Normalize the weights of the final linear layer
    normalize_weights(final_layer)

    # Apply the final linear layer to the normalized embeddings
    final_embeddings = [final_layer(torch.tensor(embedding)).detach().numpy() for embedding in embeddings]

    # Save embeddings
    embeddings_path = os.path.join(path_root_data, 'image_embeddings.npy')
    np.save(embeddings_path, final_embeddings)
    print(f'Embeddings saved to {embeddings_path}')

    return final_embeddings

if __name__ == '__main__':
    path_root_data = os.path.join('..', 'data', 'example')
    final_embeddings = run_vit_embedding(path_root_data)

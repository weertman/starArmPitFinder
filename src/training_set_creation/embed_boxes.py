import os
import torch
from torch import nn, optim
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from create_bboxs_from_cvat11 import create_bboxs_from_cvat11
from transformers import ViTModel, ViTImageProcessor
import matplotlib.pyplot as plt

# Define the decoder
class ImageDecoder(nn.Module):
    def __init__(self, embedding_dim, img_size=(224, 224)):
        super(ImageDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, img_size[0] * img_size[1] * 3),
            nn.Sigmoid()
        )
        self.img_size = img_size

    def forward(self, x):
        x = self.decoder(x)
        x = x.view(-1, 3, self.img_size[0], self.img_size[1])
        return x

# Load ViT model and image processor
def load_vit_model(model_name='google/vit-large-patch16-224-in21k', unfreeze_layers=64):
    image_processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the top 'unfreeze_layers' layers
    for layer in model.encoder.layer[-unfreeze_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    return model, image_processor

# Preprocess images
def preprocess_image(image_path, image_processor):
    image = Image.open(image_path).convert('RGB')
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs['pixel_values']

# Fine-tune with reconstruction
def fine_tune_with_reconstruction(path_root_data, unfreeze_layers=64,
                                  final_embedding_dim=256, epochs=1, lr=1e-3):
    # Load data
    crop_image_paths, _ = create_bboxs_from_cvat11(path_root_data)
    model, image_processor = load_vit_model(unfreeze_layers=unfreeze_layers)

    # Learnable linear layer for dimensionality reduction
    linear_layer = nn.Linear(model.config.hidden_size, final_embedding_dim)
    linear_layer = nn.Sequential(linear_layer, nn.GELU())
    if torch.cuda.is_available():
        linear_layer.to('cuda')

    # Decoder for reconstruction
    decoder = ImageDecoder(embedding_dim=final_embedding_dim)
    if torch.cuda.is_available():
        decoder.to('cuda')

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        list(model.parameters()) + list(linear_layer.parameters()) + list(decoder.parameters()),
        lr=lr, momentum=0.9
    )

    # Training loop
    model.train()
    linear_layer.train()
    losses = {}
    for epoch in range(epochs):
        epoch_loss = 0
        losses[epoch] = []
        pbar = tqdm(crop_image_paths, desc=f'Epoch {epoch + 1}/{epochs}')
        for image_path in pbar:
            image_tensor = preprocess_image(image_path, image_processor)
            if torch.cuda.is_available():
                image_tensor = image_tensor.to('cuda')

            # Forward pass
            embedding = model(image_tensor).last_hidden_state[:, 0, :]
            embedding = linear_layer(embedding)

            reconstruction = decoder(embedding)
            loss = criterion(reconstruction, image_tensor)
            losses[epoch].append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / (len(losses[epoch]) + 1e-7)})

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(crop_image_paths)}')

    # Save the fine-tuned model and decoder
    torch.save(model.state_dict(), os.path.join(path_root_data, 'vit_finetuned.pth'))
    torch.save(linear_layer.state_dict(), os.path.join(path_root_data, 'linear_layer.pth'))
    torch.save(decoder.state_dict(), os.path.join(path_root_data, 'decoder.pth'))
    print('Model fine-tuned and saved.')

    # Plot the loss
    fig, ax = plt.subplots(figsize=(6, 4))
    x = 0
    for epoch, loss_values in losses.items():
        ax.plot(range(x, x + len(loss_values)), loss_values, color='blue', alpha=0.3)
        x += len(loss_values)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Reconstruction Loss')

    path_fig = os.path.join(path_root_data, 'reconstruction_loss_plot.png')
    fig.savefig(path_fig, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def embed_images_with_finetuned_model(path_root_data, final_embedding_dim=256):
    model, image_processor = load_vit_model()
    model.load_state_dict(torch.load(os.path.join(path_root_data, 'vit_finetuned.pth')))
    model.eval()

    # Load the trained linear layer
    linear_layer = nn.Sequential(
        nn.Linear(model.config.hidden_size, final_embedding_dim),
        nn.GELU()
    )
    state_dict = torch.load(os.path.join(path_root_data, 'linear_layer.pth'))
    linear_layer.load_state_dict(state_dict)
    linear_layer.eval()
    if torch.cuda.is_available():
        linear_layer.to('cuda')

    crop_image_paths, _ = create_bboxs_from_cvat11(path_root_data)
    embeddings = []

    for image_path in tqdm(crop_image_paths, desc='Embedding images'):
        image_tensor = preprocess_image(image_path, image_processor)
        if torch.cuda.is_available():
            image_tensor = image_tensor.to('cuda')

        with torch.no_grad():
            embedding = model(image_tensor).last_hidden_state[:, 0, :]
            embedding = linear_layer(embedding)
            embeddings.append(embedding.cpu().numpy())

    embeddings = np.array(embeddings)
    embeddings_path = os.path.join(path_root_data, 'image_embeddings_finetuned.npy')
    np.save(embeddings_path, embeddings)
    print(f'Embeddings saved to {embeddings_path}')
    return embeddings

def run_vit_embedding(path_root_data, final_embedding_dim=256, unfreeze_layers=32, epochs=2, lr=1e-3):
    fine_tune_with_reconstruction(path_root_data=path_root_data,
                                  unfreeze_layers=unfreeze_layers,
                                  final_embedding_dim=final_embedding_dim,
                                  epochs=epochs, lr=lr)
    embeddings = embed_images_with_finetuned_model(path_root_data, final_embedding_dim)
    return embeddings

if __name__ == '__main__':
    path_root_data = os.path.join('..', 'data', 'example')
    run_vit_embedding(path_root_data)

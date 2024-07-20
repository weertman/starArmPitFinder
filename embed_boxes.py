import os
import torch
from torch import nn, optim
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from transformers import ViTModel, ViTImageProcessor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Define the combined loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        # Move SSIM metric to the same device as the input
        self.ssim.to(output.device)

        mse = self.mse_loss(output, target)
        l1 = self.l1_loss(output, target)
        ssim_value = 1 - self.ssim(output, target)
        return self.alpha * mse + (1 - self.alpha) * l1 + self.beta * ssim_value
# Define the decoder
class ImageDecoder(nn.Module):
    def __init__(self, embedding_dim, img_size=(224, 224)):
        super(ImageDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Linear(4096, img_size[0] * img_size[1] * 3),
            nn.Tanh()  # Use Tanh if inputs are normalized to [-1, 1]
        )
        self.img_size = img_size
        self.initialize_weights()

    def forward(self, x):
        x = self.decoder(x)
        x = x.view(-1, 3, self.img_size[0], self.img_size[1])
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# Load ViT model and image processor
def load_vit_model(model_name='google/vit-large-patch16-224-in21k'):
    print(f'Loading model {model_name}')
    image_processor = ViTImageProcessor.from_pretrained(model_name, do_rescale=False)
    model = ViTModel.from_pretrained(model_name)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    if torch.cuda.is_available():
        model.to('cuda')
    return model, image_processor

# Preprocess images
def preprocess_image(image_path, image_processor):
    image = Image.open(image_path).convert('RGB')
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs['pixel_values']

# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, image_paths, image_processor, transform=None):
        self.image_paths = image_paths
        self.image_processor = image_processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)  # Remove batch dimension


# Fine-tune with reconstruction
def fine_tune_with_reconstruction(path_root_data, unfreeze_layers=64,
                                  final_embedding_dim=256, epochs=1,
                                  lr_adam=1e-3, lr_sgd=1e-3, batch_size=32, unfreeze_epoch=5,
                                  gradient_accumulation_steps=4):
    # Load data
    path_create = os.path.join(path_root_data, 'create.csv')
    df = pd.read_csv(path_create)
    crop_image_paths = df['image_path'].tolist()
    model, image_processor = load_vit_model()

    # Define augmentations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor()
    ])

    print()
    print('Unfreezing layers:', unfreeze_layers)
    print('Final embedding dimension:', final_embedding_dim)
    print('Epochs:', epochs)
    print('Learning rate (Adam):', lr_adam)
    print('Learning rate (SGD):', lr_sgd)
    print('Batch size:', batch_size)
    print('Unfreeze epoch:', unfreeze_epoch)
    print('Gradient accumulation steps:', gradient_accumulation_steps)

    # Dataset and DataLoader
    dataset = ImageDataset(crop_image_paths, image_processor, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Learnable linear layer for dimensionality reduction
    linear_layer = nn.Sequential(
        nn.Linear(model.config.hidden_size, final_embedding_dim),
        nn.GELU()
    )
    if torch.cuda.is_available():
        linear_layer.to('cuda')

    # Decoder for reconstruction
    decoder = ImageDecoder(embedding_dim=final_embedding_dim)
    if torch.cuda.is_available():
        decoder.to('cuda')

    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.5, beta=0.1)
    optimizer = optim.Adam(
        list(linear_layer.parameters()) + list(decoder.parameters()),
        lr=lr_adam
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Training loop
    model.train()
    linear_layer.train()
    decoder.train()
    losses = {}
    for epoch in range(epochs):
        epoch_loss = 0
        losses[epoch] = []
        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}')

        # Unfreeze the ViT model after `unfreeze_epoch`
        if epoch == unfreeze_epoch:
            print(f'Unfreezing layers from {unfreeze_layers} onwards')
            for layer in model.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

            optimizer = optim.SGD(
                list(model.parameters()) + list(linear_layer.parameters()) + list(decoder.parameters()),
                lr=lr_sgd, momentum=0.9
            )

        for image_tensor in pbar:
            if torch.cuda.is_available():
                image_tensor = image_tensor.to('cuda')

            # Forward pass
            embedding = model(image_tensor).last_hidden_state[:, 0, :]
            embedding = linear_layer(embedding)
            reconstruction = decoder(embedding)

            # Ensure image_tensor and reconstruction have the same shape
            loss = criterion(reconstruction, image_tensor)
            losses[epoch].append(loss.item())

            optimizer.zero_grad()
            loss = criterion(reconstruction, image_tensor)
            loss = loss / gradient_accumulation_steps  # Normalize the loss
            loss.backward()

            batch_idx = len(losses[epoch])
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Clear CUDA cache
            torch.cuda.empty_cache()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / (len(losses[epoch]) + 1)})

    # Save the fine-tuned model and decoder
    torch.save(model.state_dict(), os.path.join(path_root_data, 'vit_finetuned.pth'))
    torch.save(linear_layer.state_dict(), os.path.join(path_root_data, 'linear_layer.pth'))
    torch.save(decoder.state_dict(), os.path.join(path_root_data, 'decoder.pth'))
    print('Model fine-tuned and saved.')

    # Plot the loss
    fig, ax = plt.subplots(figsize=(6, 4))
    x = 0
    avg_losses = []
    xvals = []
    for epoch, loss_values in losses.items():
        ax.plot(range(x, x + len(loss_values)), loss_values, color='blue', alpha=0.3)
        x += len(loss_values)
        avg_loss = np.mean(loss_values)
        ax.scatter(x, avg_loss, color='blue')
        avg_losses.append(avg_loss)
        xvals.append(x)

        if epoch == unfreeze_epoch:
            ax.scatter(x, avg_loss, color='red', zorder=5)

    ax.plot(xvals, avg_losses, color='blue', label='Average Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('log(Loss)')
    ax.set_yscale('log')
    ax.set_title('Reconstruction Loss')

    path_fig = os.path.join(path_root_data, 'reconstruction_loss_plot.png')
    fig.savefig(path_fig, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    # Show a reconstruction example
    sample_image = random.choice(crop_image_paths)
    image_tensor = preprocess_image(sample_image, image_processor)
    if torch.cuda.is_available():
        image_tensor = image_tensor.to('cuda')

    # Forward pass through the model
    with torch.no_grad():
        embedding = model(image_tensor).last_hidden_state[:, 0, :]
        embedding = linear_layer(embedding)
        reconstruction = decoder(embedding)
        reconstruction = reconstruction.cpu().detach().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5

    # Normalize images to the range [0, 1]
    sample_image = image_tensor[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    sample_image = (sample_image - sample_image.min()) / (sample_image.max() - sample_image.min())
    reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(sample_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(reconstruction[0])
    ax[1].set_title('Reconstructed Image')
    ax[1].axis('off')

    path_fig = os.path.join(path_root_data, 'reconstruction_example.png')
    fig.tight_layout()
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

    path_create = os.path.join(path_root_data, 'create.csv')
    df = pd.read_csv(path_create)
    crop_image_paths = df['image_path'].tolist()
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
    ## current shape is (n_images, 1, n_features), change shape to (n_images, n_features)
    embeddings = embeddings.squeeze(1)

    print('Embeddings shape:', embeddings.shape)
    embeddings_path = os.path.join(path_root_data, 'image_embeddings_finetuned.npy')
    np.save(embeddings_path, embeddings)
    print(f'Embeddings saved to {embeddings_path}')
    return embeddings

def run_vit_embedding(path_root_data, final_embedding_dim=256, unfreeze_layers=64,
                      epochs=2, lr_adam=1e-3, lr_sgd=1e-2, batch_size=32, unfreeze_epoch=1,
                      gradient_accumulation_steps=4):
    fine_tune_with_reconstruction(path_root_data=path_root_data,
                                  unfreeze_layers=unfreeze_layers,
                                  final_embedding_dim=final_embedding_dim,
                                  epochs=epochs, lr_adam=lr_adam, lr_sgd=lr_sgd,
                                  batch_size=batch_size, unfreeze_epoch=unfreeze_epoch,
                                  gradient_accumulation_steps=gradient_accumulation_steps)
    embeddings = embed_images_with_finetuned_model(path_root_data, final_embedding_dim)
    return embeddings

if __name__ == '__main__':
    path_root_data = os.path.join('..', '..', 'data', 'example')
    run_vit_embedding(path_root_data)

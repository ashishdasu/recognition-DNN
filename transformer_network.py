# Ashish Dasu
# CS5330 — Project 5: Recognition using Deep Networks
# Vision Transformer for MNIST digit recognition. Based on the provided NetTransformer template.
# Splits images into overlapping patches, encodes them with transformer layers,
# and classifies via a learned token representation.

# import statements
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt


# Configuration for the transformer network — holds all hyperparameters.
# Defaults match the provided template for baseline results on MNIST.
class NetConfig:

    def __init__(self,
                 name='vit_base',
                 dataset='mnist',
                 patch_size=4,
                 stride=2,
                 embed_dim=48,
                 depth=4,
                 num_heads=8,
                 mlp_dim=128,
                 dropout=0.1,
                 use_cls_token=False,
                 epochs=15,
                 batch_size=64,
                 lr=1e-3,
                 weight_decay=1e-4,
                 seed=0,
                 optimizer='adamw',
                 device='mps',
                 ):

        # fixed dataset attributes
        self.image_size = 28
        self.in_channels = 1
        self.num_classes = 10

        # model architecture
        self.name = name
        self.dataset = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.use_cls_token = use_cls_token

        # training
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.optimizer = optimizer
        self.device = device


# Converts a 2D image into a sequence of patch embeddings for the transformer.
# Uses nn.Unfold to extract overlapping patches, then projects each into embedding space.
class PatchEmbedding(nn.Module):

    def __init__(self, image_size, patch_size, stride, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim

        # extract patches — overlapping if stride < patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)

        # each flattened patch has this many values
        self.patch_dim = in_channels * patch_size * patch_size

        # linear projection from raw patch pixels to embedding space
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)

        self.num_patches = self._compute_num_patches()

    # computes total number of patches extracted from one image
    def _compute_num_patches(self):
        positions_per_dim = ((self.image_size - self.patch_size) // self.stride) + 1
        return positions_per_dim * positions_per_dim

    # extracts patches and projects them into token embeddings
    # input: (B, C, H, W) -> output: (B, num_patches, embed_dim)
    def forward(self, x):
        x = self.unfold(x)           # (B, patch_dim, num_patches)
        x = x.transpose(1, 2)        # (B, num_patches, patch_dim)
        x = self.proj(x)             # (B, num_patches, embed_dim)
        return x


# Vision Transformer network for image classification.
# Embeds image patches as tokens, processes them through transformer encoder layers
# with self-attention, then classifies via either mean pooling or a CLS token.
class NetTransformer(nn.Module):

    # defines all layers: patch embedding, positional encoding, transformer encoder, classifier
    def __init__(self, config):
        super(NetTransformer, self).__init__()

        # patch embedding converts image to token sequence
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        num_tokens = self.patch_embed.num_patches
        print(f'Number of tokens: {num_tokens}')

        # optional CLS token — a learnable vector prepended to the token sequence
        self.use_cls_token = config.use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            total_tokens = num_tokens + 1
        else:
            self.cls_token = None
            total_tokens = num_tokens

        # learnable positional embedding added to each token so the model knows spatial order
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, config.embed_dim))
        self.pos_dropout = nn.Dropout(config.dropout)

        # transformer encoder: stacked layers of multi-head self-attention + feedforward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.depth,
        )

        # normalization before classification
        self.norm = nn.LayerNorm(config.embed_dim)

        # classification head: linear -> GELU -> output classes
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.num_classes)
        )

    # computes a forward pass: patches -> tokens -> transformer -> classify
    # input: (B, 1, 28, 28) -> output: (B, num_classes) log-probabilities
    def forward(self, x):
        # convert image to patch token embeddings
        x = self.patch_embed(x)

        batch_size = x.size(0)

        # prepend CLS token if used
        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        # add positional information so attention knows spatial layout
        x = x + self.pos_embed

        # dropout after embedding (regularization)
        x = self.pos_dropout(x)

        # run through the stack of transformer encoder layers
        x = self.encoder(x)

        # aggregate tokens into a single vector for classification
        if self.use_cls_token:
            x = x[:, 0]              # use the CLS token output
        else:
            x = x.mean(dim=1)        # average all token outputs

        # normalize and classify
        x = self.norm(x)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


# Trains the transformer for one epoch. Returns average loss.
def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(data)
        total += len(data)

    return total_loss / total


# Evaluates the transformer on a dataset. Returns loss and accuracy.
def test_epoch(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(data)

    return total_loss / total, 100.0 * correct / total


# main function — builds transformer with default config, trains on MNIST, reports results
def main(argv):
    os.makedirs('results', exist_ok=True)

    # use default configuration from the template
    config = NetConfig()

    # select device: MPS (Apple Silicon), CUDA, or CPU
    if config.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # reproducibility
    torch.manual_seed(config.seed)

    # load MNIST
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # build transformer model
    model = NetTransformer(config).to(device)
    print(model)

    # optimizer
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

    # train and evaluate each epoch
    train_losses = []
    test_losses = []
    test_accs = []

    print(f'\nTraining transformer for {config.epochs} epochs...')
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'  Epoch {epoch:2d}/{config.epochs}  '
              f'Train Loss: {train_loss:.4f}  '
              f'Test Loss: {test_loss:.4f}  '
              f'Test Acc: {test_acc:.2f}%')

    # plot training results
    epochs = range(1, config.epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, train_losses, 'b-o', label='Train loss')
    ax1.plot(epochs, test_losses, 'r-o', label='Test loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Transformer — Training and Testing Loss')
    ax1.legend()

    ax2.plot(epochs, test_accs, 'g-o', label='Test accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Transformer — Test Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results/transformer_results.png', dpi=150)
    print('\nTransformer results plot saved to results/transformer_results.png')

    # save metrics for the report
    with open('results/transformer_metrics.txt', 'w') as f:
        f.write(f'Config: patch={config.patch_size}, stride={config.stride}, '
                f'dim={config.embed_dim}, depth={config.depth}, heads={config.num_heads}, '
                f'mlp={config.mlp_dim}, dropout={config.dropout}, cls={config.use_cls_token}\n')
        f.write(f'Device: {device}\n')
        f.write(f'Final test accuracy: {test_accs[-1]:.2f}%\n')
        f.write(f'Best test accuracy: {max(test_accs):.2f}% (epoch {test_accs.index(max(test_accs))+1})\n\n')
        f.write('Epoch  Train_Loss  Test_Loss  Test_Acc\n')
        for i in range(config.epochs):
            f.write(f'{i+1:5d}  {train_losses[i]:.4f}      {test_losses[i]:.4f}     {test_accs[i]:.2f}%\n')
    print('Transformer metrics saved to results/transformer_metrics.txt')

    # save model
    torch.save(model.state_dict(), 'transformer_model.pth')
    print('Model saved to transformer_model.pth')


if __name__ == "__main__":
    main(sys.argv)

# Ashish Dasu (adasu)
# CS5330 — Project 5: Recognition using Deep Networks
# Extension: replaces the learned conv1 filters with a fixed Gabor filter bank
# and retrains only the remaining layers. Compares accuracy to the fully-learned CNN
# to see whether hand-crafted features can match learned ones.

# import statements
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from train_network import MyNetwork


# Generates a single Gabor filter kernel as a numpy array.
# Parameters control the orientation, frequency, and shape of the filter.
def make_gabor_kernel(ksize=5, sigma=1.0, theta=0.0, lambd=4.0, gamma=0.5, psi=0.0):
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    for y in range(ksize):
        for x in range(ksize):
            xp = (x - center) * math.cos(theta) + (y - center) * math.sin(theta)
            yp = -(x - center) * math.sin(theta) + (y - center) * math.cos(theta)
            kernel[y, x] = math.exp(-(xp**2 + gamma**2 * yp**2) / (2 * sigma**2)) * \
                           math.cos(2 * math.pi * xp / lambd + psi)
    return kernel


# Creates a bank of 10 Gabor filters at different orientations and frequencies.
# Returns a tensor of shape [10, 1, 5, 5] matching the conv1 weight format.
def create_gabor_bank():
    filters = []
    # 5 orientations × 2 frequencies = 10 filters
    orientations = [0, math.pi/5, 2*math.pi/5, 3*math.pi/5, 4*math.pi/5]
    frequencies = [3.0, 5.0]

    for lambd in frequencies:
        for theta in orientations:
            kernel = make_gabor_kernel(ksize=5, sigma=1.0, theta=theta,
                                       lambd=lambd, gamma=0.5, psi=0.0)
            filters.append(kernel)

    bank = np.array(filters).reshape(10, 1, 5, 5)
    return torch.tensor(bank, dtype=torch.float32)


# CNN with a frozen Gabor filter bank as the first layer.
# Only the layers after conv1 are trained.
class GaborNetwork(nn.Module):

    # initializes the network with fixed Gabor filters in conv1
    def __init__(self, gabor_bank):
        super(GaborNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # load Gabor filters into conv1 and freeze them
        with torch.no_grad():
            self.conv1.weight.copy_(gabor_bank)
            self.conv1.bias.zero_()
        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False

    # forward pass — identical structure to MyNetwork
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# Trains the model for one epoch. Returns average loss.
def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    total = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(data)
        total += len(data)
    return total_loss / total


# Evaluates the model. Returns loss and accuracy.
def test_epoch(model, test_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(data)
    return total_loss / total, 100.0 * correct / total


# Visualizes the Gabor filter bank alongside the learned conv1 filters for comparison.
def plot_filter_comparison(gabor_bank, learned_weights):
    fig, axes = plt.subplots(2, 10, figsize=(20, 5))

    for i in range(10):
        axes[0][i].imshow(gabor_bank[i, 0].numpy(), cmap='gray')
        axes[0][i].set_title(f'Gabor {i}', fontsize=8)
        axes[0][i].set_xticks([])
        axes[0][i].set_yticks([])

        axes[1][i].imshow(learned_weights[i, 0].detach().numpy(), cmap='gray')
        axes[1][i].set_title(f'Learned {i}', fontsize=8)
        axes[1][i].set_xticks([])
        axes[1][i].set_yticks([])

    axes[0][0].set_ylabel('Gabor\n(fixed)', fontsize=10)
    axes[1][0].set_ylabel('Learned\n(trained)', fontsize=10)

    plt.suptitle('Gabor Filter Bank vs Learned Conv1 Filters', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/gabor_vs_learned_filters.png', dpi=150)
    print('Filter comparison saved to results/gabor_vs_learned_filters.png')


# main function — trains Gabor-based network and compares to the fully-learned CNN
def main(argv):
    os.makedirs('results', exist_ok=True)

    # load MNIST
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # create Gabor filter bank
    gabor_bank = create_gabor_bank()
    print('Created Gabor filter bank: 5 orientations × 2 frequencies = 10 filters')

    # build and train the Gabor-based network
    gabor_model = GaborNetwork(gabor_bank)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, gabor_model.parameters()),
                          lr=0.01, momentum=0.5)

    print('\nTraining Gabor network (conv1 frozen, rest trainable)...')
    gabor_losses = []
    gabor_accs = []
    n_epochs = 5

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(gabor_model, train_loader, optimizer)
        test_loss, test_acc = test_epoch(gabor_model, test_loader)
        gabor_losses.append(test_loss)
        gabor_accs.append(test_acc)
        print(f'  Epoch {epoch}  Train Loss: {train_loss:.4f}  '
              f'Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%')

    # load the fully-learned model for comparison
    learned_model = MyNetwork()
    learned_model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
    learned_model.eval()
    _, learned_acc = test_epoch(learned_model, test_loader)

    # compare results
    print(f'\n--- Results Comparison ---')
    print(f'Gabor network (fixed conv1):   {gabor_accs[-1]:.2f}%')
    print(f'Learned network (trained conv1): {learned_acc:.2f}%')
    diff = learned_acc - gabor_accs[-1]
    print(f'Difference: {diff:+.2f}% ({"learned wins" if diff > 0 else "Gabor wins"})')

    # visualize filter comparison
    plot_filter_comparison(gabor_bank, learned_model.conv1.weight)

    # plot Gabor training curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, n_epochs + 1)
    ax1.plot(epochs, gabor_losses, 'b-o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Loss')
    ax1.set_title('Gabor Network — Test Loss')

    ax2.plot(epochs, gabor_accs, 'g-o')
    ax2.axhline(y=learned_acc, color='r', linestyle='--', label=f'Learned CNN ({learned_acc:.1f}%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Gabor vs Learned — Test Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results/gabor_training_results.png', dpi=150)
    print('Training results saved to results/gabor_training_results.png')

    # save metrics
    with open('results/gabor_metrics.txt', 'w') as f:
        f.write('Gabor Filter Bank Experiment\n')
        f.write(f'Gabor filters: 5 orientations × 2 frequencies = 10 filters (5x5)\n')
        f.write(f'Conv1 frozen, all other layers trained for {n_epochs} epochs\n\n')
        f.write(f'Gabor final test accuracy:   {gabor_accs[-1]:.2f}%\n')
        f.write(f'Learned final test accuracy:  {learned_acc:.2f}%\n')
        f.write(f'Difference: {diff:+.2f}%\n\n')
        for i in range(n_epochs):
            f.write(f'Epoch {i+1}: Loss={gabor_losses[i]:.4f}  Acc={gabor_accs[i]:.2f}%\n')
    print('Metrics saved to results/gabor_metrics.txt')


if __name__ == "__main__":
    main(sys.argv)

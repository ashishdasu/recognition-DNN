# Ashish Dasu (adasu)
# CS5330 — Project 5: Recognition using Deep Networks
# Builds, trains, and saves a convolutional neural network for MNIST digit recognition.

# import statements
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt


# Convolutional neural network for classifying 28x28 grayscale digit images.
# Two convolutional layers extract spatial features (edges, curves, shapes),
# followed by max pooling to reduce dimensions and two fully connected layers
# that map the learned features to 10 digit classes.
class MyNetwork(nn.Module):

    # defines the layers of the network
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)       # 1x28x28 -> 10x24x24
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)      # 10x12x12 -> 20x8x8
        self.conv2_drop = nn.Dropout2d(p=0.5)              # randomly zeros channels to reduce overfitting
        self.fc1 = nn.Linear(320, 50)                       # 20*4*4 flattened -> 50 hidden nodes
        self.fc2 = nn.Linear(50, 10)                        # 50 -> 10 digit classes

    # computes a forward pass for the network
    # applies convolutions with pooling and ReLU, then classifies via fully connected layers
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))                     # conv1 -> 2x2 max pool -> relu
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))    # conv2 -> dropout -> 2x2 max pool -> relu
        x = x.view(-1, 320)                                             # flatten feature maps to vector
        x = F.relu(self.fc1(x))                                         # hidden layer with relu activation
        x = F.log_softmax(self.fc2(x), dim=1)                           # output log-probabilities for NLL loss
        return x


# Trains the model for one complete pass through the training data.
# Returns average loss, accuracy, and a list of (example_count, loss) pairs for plotting.
def train_network(model, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # accumulate stats for epoch-level metrics
        total_loss += loss.item() * len(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += len(data)

        if batch_idx % log_interval == 0:
            examples_seen = (epoch - 1) * len(train_loader.dataset) + batch_idx * len(data)
            losses.append((examples_seen, loss.item()))
            print(f'  Epoch {epoch} [{batch_idx * len(data):5d}/{len(train_loader.dataset)}]  '
                  f'Loss: {loss.item():.4f}')

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, losses


# Evaluates the model on a dataset without updating weights.
# Uses torch.no_grad() to disable gradient computation during evaluation.
# Returns average loss and accuracy percentage.
def test_network(model, test_loader):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    total = len(test_loader.dataset)
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# Displays the first 6 examples from the test set as a 2x3 subplot grid.
# Uses unnormalized images so pixel values render correctly in grayscale.
def plot_first_six(test_dataset):
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    for i, ax in enumerate(axes.flat):
        image, label = test_dataset[i]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('First 6 MNIST Test Examples')
    plt.tight_layout()
    plt.savefig('results/first_six_test.png', dpi=150)



# Plots training vs testing loss and accuracy curves across epochs.
# Generates two separate figures: one for loss comparison, one for accuracy.
def plot_training_results(train_losses, test_losses, train_accs, test_accs):
    epochs = range(1, len(train_losses) + 1)

    # negative log-likelihood loss per epoch
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, 'b-o', label='Train loss')
    ax.plot(epochs, test_losses, 'r-o', label='Test loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Negative Log Likelihood Loss')
    ax.set_title('Training and Testing Loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/loss_plot.png', dpi=150)


    # classification accuracy per epoch
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_accs, 'b-o', label='Train accuracy')
    ax.plot(epochs, test_accs, 'r-o', label='Test accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training and Testing Accuracy')
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/accuracy_plot.png', dpi=150)



# main function — loads MNIST data, trains the CNN for 5 epochs, and saves the model
def main(argv):
    # training hyperparameters
    n_epochs = 5
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.5

    # standard MNIST normalization values (precomputed across the dataset)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    # load MNIST training set (60k images) and test set (10k images)
    train_dataset = torchvision.datasets.MNIST('data', train=True, download=True,
                                                transform=transform)
    test_dataset = torchvision.datasets.MNIST('data', train=False, download=True,
                                               transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000,
                                               shuffle=False)

    # unnormalized test set for visualization (pixel values in [0,1] for imshow)
    test_dataset_raw = torchvision.datasets.MNIST('data', train=False, download=True,
                                                   transform=torchvision.transforms.ToTensor())

    os.makedirs('results', exist_ok=True)

    # show what the data looks like before training
    print('Plotting first 6 test examples...')
    plot_first_six(test_dataset_raw)

    # initialize network and optimizer
    model = MyNetwork()
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # train for n_epochs, evaluating on both sets after each epoch to track generalization
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(1, n_epochs + 1):
        print(f'\nEpoch {epoch}/{n_epochs}')
        train_loss, train_acc, _ = train_network(model, train_loader, optimizer, epoch)
        test_loss, test_acc = test_network(model, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f'  Train — Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'  Test  — Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')

    # visualize how loss and accuracy evolved during training
    print('\nPlotting training results...')
    plot_training_results(train_losses, test_losses, train_accs, test_accs)

    # save training metrics to file for later review and report
    with open('results/training_metrics.txt', 'w') as f:
        f.write('Epoch  Train_Loss  Test_Loss  Train_Acc  Test_Acc\n')
        for i in range(n_epochs):
            f.write(f'{i+1:5d}  {train_losses[i]:.4f}      {test_losses[i]:.4f}     '
                    f'{train_accs[i]:.2f}%    {test_accs[i]:.2f}%\n')
    print('\nTraining metrics saved to results/training_metrics.txt')

    # persist the trained weights so other scripts can load them
    torch.save(model.state_dict(), 'mnist_model.pth')
    print('Model saved to mnist_model.pth')


if __name__ == "__main__":
    main(sys.argv)

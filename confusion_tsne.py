# Ashish Dasu
# CS5330 — Project 5: Recognition using Deep Networks
# Extension: confusion matrix and t-SNE embedding visualization.
# Loads the trained CNN, runs inference on the MNIST test set, and generates
# a confusion matrix heatmap and a 2D t-SNE plot of fc1 activations.

# import statements
import sys
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from train_network import MyNetwork


# Extracts fc1 activations (50-dim) and predictions for every test image.
# Hooks into the forward pass to capture the hidden layer output before fc2.
def extract_features(model, test_loader, device):
    model.eval()
    features = []
    predictions = []
    labels = []

    # register a forward hook on fc1 to capture its output
    activation = {}
    def hook_fn(module, input, output):
        activation['fc1'] = output.detach()

    handle = model.fc1.register_forward_hook(hook_fn)

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            features.append(activation['fc1'].cpu().numpy())
            predictions.append(pred.cpu().numpy())
            labels.append(target.numpy())

    handle.remove()

    features = np.concatenate(features)
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    return features, predictions, labels


# Plots a confusion matrix heatmap showing predicted vs actual digit classes.
def plot_confusion_matrix(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=range(10), yticks=range(10),
           xlabel='Predicted', ylabel='Actual',
           title='MNIST Confusion Matrix')

    # annotate each cell with count
    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=9)

    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150)
    plt.savefig('report_images/confusion_matrix.png', dpi=150)
    print('Confusion matrix saved')

    # save misclassification summary
    total = cm.sum()
    correct = cm.trace()
    with open('results/confusion_matrix_stats.txt', 'w') as f:
        f.write(f'Overall accuracy: {100.0 * correct / total:.2f}%\n')
        f.write(f'Total correct: {correct}/{total}\n\n')
        f.write('Most confused pairs (off-diagonal):\n')
        # find top 5 confusions
        pairs = []
        for i in range(10):
            for j in range(10):
                if i != j and cm[i, j] > 0:
                    pairs.append((cm[i, j], i, j))
        pairs.sort(reverse=True)
        for count, actual, pred in pairs[:10]:
            f.write(f'  {actual} misclassified as {pred}: {count} times\n')
    print('Confusion stats saved to results/confusion_matrix_stats.txt')
    return cm


# Runs t-SNE on the 50-dim fc1 features and plots a 2D scatter plot colored by digit class.
def plot_tsne(features, labels):
    print('Running t-SNE on 10000 samples (this may take ~30s)...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                         c=labels, cmap='tab10', s=3, alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label('Digit Class')
    ax.set_title('t-SNE of CNN Hidden Layer (fc1) Activations')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig('results/tsne_embedding.png', dpi=150)
    plt.savefig('report_images/tsne_embedding.png', dpi=150)
    print('t-SNE plot saved')


# main function
def main(argv):
    # load trained model
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))

    device = torch.device('cpu')  # CPU is fine for inference
    model = model.to(device)

    # load MNIST test set
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST('data', train=False, download=True,
                                               transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000,
                                               shuffle=False)

    # extract features and predictions
    print('Extracting fc1 features from test set...')
    features, predictions, labels = extract_features(model, test_loader, device)
    print(f'Extracted {len(features)} feature vectors (dim={features.shape[1]})')

    # confusion matrix
    plot_confusion_matrix(labels, predictions)

    # t-SNE embedding
    plot_tsne(features, labels)

    print('Done.')


if __name__ == "__main__":
    main(sys.argv)

# Ashish Dasu
# CS5330 — Project 5: Recognition using Deep Networks
# Re-uses the trained MNIST digit CNN to classify Greek letters via transfer learning.
# Freezes the pre-trained convolutional layers and retrains only the final classification layer.

# import statements
import os
import sys
import glob
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from train_network import MyNetwork


# Transforms color Greek letter images to match MNIST format:
# grayscale, scaled and cropped to 28x28, intensity-inverted (white strokes on black).
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# Trains the modified network on the Greek letter dataset for one epoch.
# Returns the average loss over all batches.
def train_greek(model, train_loader, optimizer):
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


# Evaluates the model on a data loader and returns loss and accuracy.
# Runs in no-grad mode since we only need forward pass results.
def test_greek(model, data_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(data)

    if total == 0:
        return 0, 0
    return total_loss / total, 100.0 * correct / total


# Loads and preprocesses a single Greek letter image for classification.
# Applies the same pipeline as GreekTransform: grayscale, scale, crop, invert, normalize.
def load_greek_image(path):
    img = Image.open(path).convert('RGB')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        GreekTransform(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(img).unsqueeze(0)


# Classifies all Greek letter test images and displays results.
# Reads images from subdirectories of test_dir (folder name = true label).
def evaluate_greek_test(model, test_dir, class_names):
    model.eval()
    images = []
    true_labels = []
    predictions = []
    filenames = []

    for class_idx, name in enumerate(class_names):
        class_dir = os.path.join(test_dir, name)
        if not os.path.isdir(class_dir):
            continue
        for path in sorted(glob.glob(os.path.join(class_dir, '*.png'))):
            tensor = load_greek_image(path)
            with torch.no_grad():
                output = model(tensor)
            pred = output.argmax().item()

            images.append(tensor.squeeze().numpy())
            true_labels.append(class_idx)
            predictions.append(pred)
            filenames.append(os.path.basename(path))

    if not images:
        print(f'\nNo test images found in {test_dir}/ — skipping.')
        return

    # print classification results
    print(f'\n--- Greek Letter Test Results ({len(images)} images) ---')
    print(f'{"File":<25}  {"Pred":<10}  {"True":<10}  {"Match"}')
    print('-' * 55)
    correct = 0
    for i in range(len(images)):
        pred_name = class_names[predictions[i]] if predictions[i] < len(class_names) else '?'
        true_name = class_names[true_labels[i]]
        match = '✓' if predictions[i] == true_labels[i] else '✗'
        if predictions[i] == true_labels[i]:
            correct += 1
        print(f'{filenames[i]:<25}  {pred_name:<10}  {true_name:<10}  {match}')

    print(f'\nAccuracy: {correct}/{len(images)} ({100*correct/len(images):.0f}%)')

    # plot results in a grid
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    axes_flat = np.array(axes).reshape(-1) if n > 1 else [axes]

    for i, ax in enumerate(axes_flat):
        if i < n:
            display = images[i] * 0.3081 + 0.1307
            ax.imshow(display, cmap='gray')
            pred_name = class_names[predictions[i]] if predictions[i] < len(class_names) else '?'
            true_name = class_names[true_labels[i]]
            color = 'green' if predictions[i] == true_labels[i] else 'red'
            ax.set_title(f'Pred: {pred_name}\nTrue: {true_name}', fontsize=10, color=color)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Greek Letter Classification Results', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/greek_test_results.png', dpi=150)
    print('Greek test results saved to results/greek_test_results.png')


# main function — sets up transfer learning, trains on Greek letters, evaluates
def main(argv):
    os.makedirs('results', exist_ok=True)

    # detect available classes from greek_train/ subdirectories
    training_set_path = 'greek_train'
    class_names = sorted([d for d in os.listdir(training_set_path)
                          if os.path.isdir(os.path.join(training_set_path, d))])
    num_classes = len(class_names)
    print(f'Detected {num_classes} classes: {class_names}')

    # load pre-trained MNIST network
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))

    # freeze all existing network weights so only the new layer trains
    for param in model.parameters():
        param.requires_grad = False

    # replace the final classification layer to match the number of Greek letter classes
    model.fc2 = torch.nn.Linear(50, num_classes)
    print('\nModified network (fc2 replaced):')
    print(model)

    # set up the Greek letter training data with the prescribed transform pipeline
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=5,
        shuffle=True
    )

    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01, momentum=0.9)
    n_epochs = 200
    train_losses = []

    print(f'\nTraining on {num_classes} Greek letter classes...')
    for epoch in range(1, n_epochs + 1):
        loss = train_greek(model, greek_train, optimizer)
        train_losses.append(loss)
        _, acc = test_greek(model, greek_train)

        if epoch % 25 == 0 or epoch <= 5 or acc == 100.0:
            print(f'  Epoch {epoch:3d}  Loss: {loss:.4f}  Train Acc: {acc:.1f}%')

        # stop early if we've achieved perfect training accuracy
        if acc == 100.0:
            print(f'\nReached 100% training accuracy at epoch {epoch}.')
            break

    # plot training loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, 'b-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Greek Letter Transfer Learning — Training Loss')
    plt.tight_layout()
    plt.savefig('results/greek_training_loss.png', dpi=150)
    print('Training loss plot saved to results/greek_training_loss.png')

    # save training metrics for the report
    with open('results/greek_training_metrics.txt', 'w') as f:
        f.write(f'Classes: {class_names}\n')
        f.write(f'Total epochs: {len(train_losses)}\n')
        f.write(f'Final loss: {train_losses[-1]:.4f}\n')
        for i, loss in enumerate(train_losses):
            f.write(f'Epoch {i+1}: {loss:.4f}\n')
    print('Training metrics saved to results/greek_training_metrics.txt')

    # save modified network structure to file for the report
    with open('results/greek_model_structure.txt', 'w') as f:
        f.write('Modified network (fc2 replaced for Greek letter classification):\n')
        f.write(str(model) + '\n')
        f.write(f'\nClasses: {class_names}\n')
        f.write(f'Total epochs trained: {len(train_losses)}\n')
        f.write(f'Final training loss: {train_losses[-1]:.4f}\n')
    print('Model structure saved to results/greek_model_structure.txt')

    # save the trained transfer learning model
    torch.save(model.state_dict(), 'greek_model.pth')
    print('Model saved to greek_model.pth')

    # evaluate on user's own Greek letter test images
    evaluate_greek_test(model, 'greek_test', class_names)


if __name__ == "__main__":
    main(sys.argv)

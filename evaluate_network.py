# Ashish Dasu (adasu)
# CS5330 — Project 5: Recognition using Deep Networks
# Loads a trained MNIST model and evaluates it on the test set and handwritten digit images.

# import statements
import os
import sys
import glob
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from train_network import MyNetwork


# Runs the first 10 test examples through the network and prints output values.
# For each example: prints all 10 log-probability outputs (2 decimal places),
# the predicted class (argmax), and the correct label.
def evaluate_test_examples(model, test_loader):
    model.eval()
    data, target = next(iter(test_loader))
    first_10_data = data[:10]
    first_10_target = target[:10]

    with torch.no_grad():
        output = model(first_10_data)

    print('\n--- First 10 Test Examples ---')
    print(f'{"Idx":>3}  {"Output Values (log-probabilities)":<65}  {"Pred":>4}  {"True":>4}')
    print('-' * 82)
    for i in range(10):
        values = output[i].numpy()
        values_str = '  '.join(f'{v:6.2f}' for v in values)
        pred = output[i].argmax().item()
        true = first_10_target[i].item()
        match = '✓' if pred == true else '✗'
        print(f'{i:3d}  {values_str}   {pred:4d}  {true:4d}  {match}')

    return first_10_data, first_10_target, output


# Plots the first 9 test digits in a 3x3 grid with the predicted label above each.
def plot_test_predictions(data, target, output):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        image = data[i].squeeze().numpy()
        pred = output[i].argmax().item()
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Prediction: {pred}', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('First 9 Test Digits with Predictions', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/test_predictions.png', dpi=150)
    print('\nTest predictions plot saved to results/test_predictions.png')


# Loads a handwritten digit image and preprocesses it to match MNIST format.
# Converts to grayscale, resizes to 28x28, inverts intensities (MNIST is white-on-black),
# and applies the same normalization used during training.
def load_handwritten_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)

    # invert: handwritten images are dark ink on light paper,
    # but MNIST digits are white strokes on black background
    img_array = 255.0 - img_array

    # normalize to [0,1] then apply MNIST normalization
    img_array = img_array / 255.0
    img_array = (img_array - 0.1307) / 0.3081

    return torch.tensor(img_array).unsqueeze(0).unsqueeze(0)


# Classifies all handwritten digit images in the given directory.
# Expects files named 0.png through 9.png (the true label is the filename).
# Displays each digit with its predicted label in a grid.
def evaluate_handwritten(model, digits_dir):
    model.eval()
    image_paths = sorted(glob.glob(os.path.join(digits_dir, '*.png')))

    if not image_paths:
        print(f'\nNo images found in {digits_dir}/ — skipping handwritten evaluation.')
        return

    images = []
    true_labels = []
    predictions = []

    print('\n--- Handwritten Digit Classification ---')
    print(f'{"File":<20}  {"Pred":>4}  {"True":>4}  {"Match"}')
    print('-' * 40)

    for path in image_paths:
        filename = os.path.basename(path)
        true_label = int(os.path.splitext(filename)[0])
        tensor = load_handwritten_image(path)

        with torch.no_grad():
            output = model(tensor)
        pred = output.argmax().item()

        images.append(tensor.squeeze().numpy())
        true_labels.append(true_label)
        predictions.append(pred)

        match = '✓' if pred == true_label else '✗'
        print(f'{filename:<20}  {pred:4d}  {true_label:4d}  {match}')

    # compute and print accuracy
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    print(f'\nAccuracy: {correct}/{len(predictions)} ({100*correct/len(predictions):.0f}%)')

    # plot all digits in a grid with predictions
    n = len(images)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 3 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes_flat = np.array(axes).flat

    for i, ax in enumerate(axes_flat):
        if i < n:
            # denormalize for display
            display = images[i] * 0.3081 + 0.1307
            ax.imshow(display, cmap='gray')
            color = 'green' if predictions[i] == true_labels[i] else 'red'
            ax.set_title(f'Pred: {predictions[i]} (True: {true_labels[i]})',
                         fontsize=10, color=color)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Handwritten Digit Classification', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/handwritten_results.png', dpi=150)
    print('Handwritten results plot saved to results/handwritten_results.png')


# main function — loads saved model and runs both test set and handwritten evaluation
def main(argv):
    os.makedirs('results', exist_ok=True)

    # load the trained model
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
    model.eval()
    print('Loaded model from mnist_model.pth')

    # load MNIST test set (not shuffled so we always get the same first 10)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST('data', train=False, download=True,
                                               transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10,
                                               shuffle=False)

    # evaluate and display first 10 test examples
    data, target, output = evaluate_test_examples(model, test_loader)
    plot_test_predictions(data, target, output)

    # save the printed output table to file for the report
    with open('results/evaluate_output.txt', 'w') as f:
        f.write('--- First 10 Test Examples ---\n')
        f.write(f'{"Idx":>3}  {"Output Values (log-probabilities)":<65}  {"Pred":>4}  {"True":>4}\n')
        f.write('-' * 82 + '\n')
        for i in range(10):
            values = output[i].detach().numpy()
            values_str = '  '.join(f'{v:6.2f}' for v in values)
            pred = output[i].argmax().item()
            true = target[i].item()
            match = '✓' if pred == true else '✗'
            f.write(f'{i:3d}  {values_str}   {pred:4d}  {true:4d}  {match}\n')
    print('Output table saved to results/evaluate_output.txt')

    # evaluate handwritten digits if the directory exists
    evaluate_handwritten(model, 'handwritten_digits')


if __name__ == "__main__":
    main(sys.argv)

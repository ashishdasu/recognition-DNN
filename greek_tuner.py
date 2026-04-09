# Ashish Dasu
# CS5330 — Project 5: Recognition using Deep Networks
# Extension: systematic hyperparameter tuning for Greek letter transfer learning.
# Tests 10 configurations across 3 freeze levels, 2 optimizers, and multiple
# learning rates to identify the best transfer learning strategy.

# import statements
import sys
import os
import glob
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from PIL import Image
from train_network import MyNetwork


# Transforms color Greek letter images to match MNIST format.
class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# Evaluates a model on pre-loaded test images. Returns (correct, total).
def evaluate(model, test_images, test_labels):
    model.eval()
    correct = 0
    with torch.no_grad():
        for img, label in zip(test_images, test_labels):
            pred = model(img).argmax().item()
            if pred == label:
                correct += 1
    return correct, len(test_labels)


# Trains and evaluates a single transfer learning configuration.
# Unfreezes layers based on the unfreeze parameter ('fc2', 'fc1+fc2', or 'all'),
# replaces fc2 with a new classification layer, and trains until 100% train
# accuracy or max_epochs. Returns best test accuracy and the epoch it occurred.
def run_config(name, unfreeze, opt_name, lr, wd, max_epochs,
               train_loader, test_images, test_labels, num_classes):
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))

    # freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # selectively unfreeze layers based on config
    if unfreeze == 'fc2':
        pass  # only the new fc2 will be trainable
    elif unfreeze == 'fc1+fc2':
        for p in model.fc1.parameters():
            p.requires_grad = True
    elif unfreeze == 'all':
        for p in model.parameters():
            p.requires_grad = True

    # replace classification layer for Greek letter classes
    model.fc2 = torch.nn.Linear(50, num_classes)

    trainable = [p for p in model.parameters() if p.requires_grad]

    if opt_name == 'sgd':
        optimizer = optim.SGD(trainable, lr=lr, momentum=0.9)
    elif opt_name == 'adam':
        optimizer = optim.Adam(trainable, lr=lr, weight_decay=wd)

    best_acc = 0
    best_epoch = 0
    for epoch in range(1, max_epochs + 1):
        # train one epoch
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = F.nll_loss(model(data), target)
            loss.backward()
            optimizer.step()

        # evaluate on test set
        correct, total = evaluate(model, test_images, test_labels)
        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        # check training accuracy for early stopping
        model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for data, target in train_loader:
                pred = model(data).argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += len(data)
        train_acc = 100.0 * train_correct / train_total

        if train_acc == 100.0 and epoch > 10:
            break

    print(f'  {name:<40s}  best_test={best_acc:.0f}% (ep {best_epoch})'
          f'  final_train={train_acc:.0f}%  epochs={epoch}')
    return best_acc, best_epoch


# main function — loads data, runs all 10 tuning configurations, ranks results
def main(argv):
    training_set_path = 'greek_train'
    class_names = sorted([d for d in os.listdir(training_set_path)
                          if os.path.isdir(os.path.join(training_set_path, d))])
    num_classes = len(class_names)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        GreekTransform(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path, transform=transform),
        batch_size=5, shuffle=True
    )

    # pre-load all test images for fast evaluation
    test_images = []
    test_labels = []
    for idx, name in enumerate(class_names):
        test_dir = os.path.join('greek_test', name)
        if not os.path.isdir(test_dir):
            continue
        for path in sorted(glob.glob(os.path.join(test_dir, '*.png'))):
            img = Image.open(path).convert('RGB')
            test_images.append(transform(img).unsqueeze(0))
            test_labels.append(idx)

    # 10 configurations: 3 freeze levels × multiple optimizer/LR combos
    configs = [
        ('fc2-only SGD lr=0.01',           'fc2',     'sgd',  0.01,   0,    100),
        ('fc2-only Adam lr=0.001',          'fc2',     'adam', 0.001,  0,    100),
        ('fc1+fc2 SGD lr=0.01',            'fc1+fc2', 'sgd',  0.01,   0,    100),
        ('fc1+fc2 Adam lr=0.001',          'fc1+fc2', 'adam', 0.001,  0,    100),
        ('fc1+fc2 Adam lr=0.0005 wd=1e-4', 'fc1+fc2', 'adam', 0.0005, 1e-4, 100),
        ('all SGD lr=0.001',               'all',     'sgd',  0.001,  0,    100),
        ('all SGD lr=0.005',               'all',     'sgd',  0.005,  0,    100),
        ('all Adam lr=0.0001',             'all',     'adam', 0.0001, 0,    100),
        ('all Adam lr=0.0005',             'all',     'adam', 0.0005, 0,    100),
        ('all Adam lr=0.0005 wd=1e-4',    'all',     'adam', 0.0005, 1e-4, 100),
    ]

    print(f'Classes: {class_names} ({num_classes})')
    print(f'Train: {len(train_loader.dataset)}, Test: {len(test_labels)}')
    print()

    results = []
    for name, unfreeze, opt, lr, wd, epochs in configs:
        acc, ep = run_config(name, unfreeze, opt, lr, wd, epochs,
                             train_loader, test_images, test_labels, num_classes)
        results.append((name, acc, ep))

    # print ranked results
    print(f'\nRANKED BY TEST ACCURACY:')
    for name, acc, ep in sorted(results, key=lambda x: -x[1]):
        print(f'  {acc:5.1f}%  {name} (epoch {ep})')

    # save to file
    os.makedirs('results', exist_ok=True)
    with open('results/greek_tuner_results.txt', 'w') as f:
        f.write(f'Classes: {class_names} ({num_classes})\n')
        f.write(f'Train: {len(train_loader.dataset)}, Test: {len(test_labels)}\n\n')
        f.write('RANKED BY TEST ACCURACY:\n')
        for name, acc, ep in sorted(results, key=lambda x: -x[1]):
            f.write(f'  {acc:5.1f}%  {name} (epoch {ep})\n')
    print('\nSaved to results/greek_tuner_results.txt')


if __name__ == "__main__":
    main(sys.argv)

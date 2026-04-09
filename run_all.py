# Ashish Dasu
# CS5330 — Project 5: Recognition using Deep Networks
# Runs all project scripts in dependency order and reports success/failure for each.
# Skips live_digit_recognition.py since it requires a webcam and manual interaction.

# import statements
import subprocess
import sys
import time


# Scripts in dependency order — each entry is (name, command, description).
# train_network.py must run first since other scripts load its saved model.
SCRIPTS = [
    ('train_network',          'python train_network.py',          'Train CNN on MNIST, save model'),
    ('evaluate_network',       'python evaluate_network.py',       'Evaluate on test set + handwritten digits'),
    ('examine_network',        'python examine_network.py',        'Visualize conv1 filters and effects'),
    ('greek_letters',          'python greek_letters.py',          'Transfer learning on Greek letters'),
    ('greek_tuner',            'python greek_tuner.py',            'Greek letter hyperparameter tuning'),
    ('transformer_network',    'python transformer_network.py',    'Train transformer on MNIST'),
    ('experiment',             'python experiment.py',             'Fashion MNIST architecture experiment'),
    ('gabor_experiment',       'python gabor_experiment.py',       'Gabor filter bank comparison'),
    ('confusion_tsne',         'python confusion_tsne.py',         'Confusion matrix and t-SNE visualization'),
    ('augmentation_experiment', 'python augmentation_experiment.py', 'Data augmentation comparison'),
]


# main function — runs each script sequentially, stops on first failure
def main(argv):
    results = []
    total_start = time.time()

    for name, cmd, desc in SCRIPTS:
        print(f'\n{"="*60}')
        print(f'RUNNING: {name} — {desc}')
        print(f'{"="*60}\n')

        start = time.time()
        result = subprocess.run(cmd, shell=True)
        elapsed = time.time() - start

        status = 'PASS' if result.returncode == 0 else 'FAIL'
        results.append((name, status, elapsed))
        print(f'\n→ {name}: {status} ({elapsed:.1f}s)')

        if result.returncode != 0:
            print(f'\n*** {name} FAILED — stopping here. Fix the error and re-run. ***')
            break

    total_time = time.time() - total_start
    print(f'\n{"="*60}')
    print(f'SUMMARY ({total_time:.0f}s total)')
    print(f'{"="*60}')
    for name, status, elapsed in results:
        print(f'  {status}  {name:<25} ({elapsed:.1f}s)')

    # note the one script not included
    print(f'\nNote: live_digit_recognition.py requires a webcam — run it manually.')


if __name__ == "__main__":
    main(sys.argv)

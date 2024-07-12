import subprocess as sp
import argparse

def main(ti, numfold)
    # run inference for ti times with numfold-fold cross-validation
    for i in range(ti):
        for n in range(numfold):
            sp.call(['python3.10', 'inference.py', '--ti', str(i), '--fold', str(n)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ti', type=int, default=5, help='Number of times to run times of inference')
    parser.add_argument('--fold', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()
    main(args.ti, args.fold)
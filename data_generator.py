import numpy as np
import argparse
import os
import data_process

parser = argparse.ArgumentParser(
    description=("Generating data (adding gaussian inputs)."), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dir', type=str, default="data/vanilla_data/",
                    help='Path to directory where model and plots will be stored.')
parser.add_argument("--S_over_B", type=str, default="",
                    help="Signal vs background ratio")
# parser.add_argument("--add_gauss", action="store_true", default=False,
#                     help="to add gaussian inputs into existing X_files")
parser.add_argument("--gaussian", type=int, default=0,
                    help="number of gaussian input")
parser.add_argument("--uniform", type=int, default=0,
                    help="number of uniform input")
parser.add_argument("--signal_vs_bg", action="store_true", default=False,
                    help="train over signal vs background")
parser.add_argument("--testset", action="store_true", default=False,
                    help="make test set: all models are evaluated on the same test set")
args = parser.parse_args()

data_train = data_process.data_loader(args.input_dir, signal_vs_bg = args.signal_vs_bg, train = True)
data_val = data_process.data_loader(args.input_dir, signal_vs_bg = args.signal_vs_bg, val = True)
data_test = data_process.data_loader(args.input_dir)

X_train, X_val, X_test, y_train, y_val, y_test = data_process.data_prep(data_train, data_val, data_test, gauss_num=args.gaussian, uniform_num=args.uniform)

if args.testset:
    np.save(os.path.join('data/X_files/X_test.npy'), X_test)
    np.save(os.path.join('data/X_files/y_test.npy'), y_test)
    print("Testset is generated.")
else:
    if args.signal_vs_bg:
        np.save(os.path.join('data/X_files/X_train_sb'+args.S_over_B+'.npy'), X_train)
        np.save(os.path.join('data/X_files/X_val_sb'+args.S_over_B+'.npy'), X_val)
        np.save(os.path.join('data/X_files/y_train_sb'+args.S_over_B+'.npy'), y_train)
        np.save(os.path.join('data/X_files/y_val_sb'+args.S_over_B+'.npy'), y_val)
        print("X_files (sb) are saved.")
    else:
        np.save(os.path.join('data/X_files/X_train'+args.S_over_B+'.npy'), X_train)
        np.save(os.path.join('data/X_files/X_val'+args.S_over_B+'.npy'), X_val)
        np.save(os.path.join('data/X_files/y_train'+args.S_over_B+'.npy'), y_train)
        np.save(os.path.join('data/X_files/y_val'+args.S_over_B+'.npy'), y_val)
        print("X_files are saved.")
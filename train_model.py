import argparse
import os
import sys
from team_code import train_models

def get_parser():
    description = 'Train the Challenge models.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-t', '--test_folder', type=str, required=True)
    parser.add_argument('-m', '--model_folder', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser

def run(args):
    train_folder = args.data_folder
    test_folder = args.test_folder
    model_folder = args.model_folder
    verbose = args.verbose

    train_models(train_folder, test_folder, model_folder, verbose)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))

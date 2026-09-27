import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import parse_cli_args, build_run_options
from train import run_experiment


def main():
    cli_args = parse_cli_args()
    run_options = build_run_options(cli_args)
    run_experiment(run_options)


if __name__ == '__main__':
    main()

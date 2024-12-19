import sys
import argparse
from . import core



def main():

    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description=f"command line interface for accuracy bases xai-evaluation",
    )

    subparsers = parser.add_subparsers(dest="cmd", help="")

    # specify different commands
    parser_a = subparsers.add_parser("pg-b-mp", help="polygon based mask processing")
    parser_a.add_argument("--mask-dir", "-md", help="directory where masks are located")

    args = parser.parse_args()


    if args.cmd == "pg-b-mp":
        core.polygon_based_mask_processing(args.mask_dir)

    else:
        print(f"unknown command: {args.cmd}")
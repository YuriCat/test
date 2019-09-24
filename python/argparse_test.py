import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()

parser.add_argument('--a')
parser.add_argument('--b', type=bool)
parser.add_argument('--c', type=strtobool)

print(vars(parser.parse_args()))
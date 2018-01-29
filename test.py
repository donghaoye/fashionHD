import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--a', action = 'store_true')

print(parser.parse_args())

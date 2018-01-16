import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--a', type = int, default = -1)

def f(parser, *argv):
    if argv:
        return parser.parse_args(argv[0])
    else:
        return parser.parse_args()

if __name__ == '__main__':
    opt = f(parser, ['--a', '0'])
    print(opt)


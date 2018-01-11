import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--a', type = bool, default = True)
parser.add_argument('--b', default = False, action = 'store_true')

def f(opt):
    opt = copy(opt)
    opt.a = False

if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)
    f(opt)
    print(opt)

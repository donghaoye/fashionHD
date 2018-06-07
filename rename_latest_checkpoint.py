import shutil
import glob
import sys

model_id = sys.argv[1]
epoch = sys.argv[2]

model_path = 'checkpoints/' + model_id + '/'
print(model_path)
for fn in glob.glob(model_path + 'latest_*.pth'):
    new_fn =fn.replace('latest', epoch)
    shutil.copyfile(fn, new_fn)
    print('%s => %s' % (fn, new_fn))

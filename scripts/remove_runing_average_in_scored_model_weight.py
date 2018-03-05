import torch
import os
import sys
import shutil


for root, dirs, files in os.walk('checkpoints'):
    for fn in files:
        if fn.startswith('latest_net_') and fn.endswith('.pth'):
            print(os.path.join(root, fn))
            fn_backup = 'backup_' + fn
            if not os.path.isfile(os.path.join(root, fn_backup)):
                shutil.copyfile(os.path.join(root, fn), os.path.join(root, fn_backup))
                state_dict = torch.load(os.path.join(root, fn))
                for k, v in state_dict.iteritems():
                    if 'running_' in k:
                        state_dict.pop(k)
                torch.save(state_dict, os.path.join(root, fn))
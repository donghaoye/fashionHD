from __future__ import division, print_function

import torch.utils.data
import copy

# Todo: disentangle data-related parameters from model options

def CreateDataLoader(opt, split = None):
    # loader = CustomDataLoader()
    # loader.initialize(opt)
    # return loader

    if split is None:
        split = 'train' if opt.is_train else 'test'

    dataset = CreateDataset(opt, split)
    shuffle = (split == 'train')
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset, 
        batch_size = opt.batch_size,
        shuffle = shuffle, 
        num_workers = int(opt.nThreads), 
        drop_last = (split == 'train' and opt.is_train),
        pin_memory = True)

    return dataloader


def CreateDataset(opt, split):
    dataset = None
    
    if opt.dataset_mode == 'attribute':
        from data.attribute_dataset import AttributeDataset
        dataset = AttributeDataset()
    elif opt.dataset_mode == 'attribute_exp':
        from data.exp_attribute_dataset import EXPAttributeDataset
        dataset = EXPAttributeDataset()
    elif opt.dataset_mode in {'gan_self'}:
        from data.gan_dataset import GANDataset
        dataset = GANDataset()
    else:
        raise ValueError('Dataset mode [%s] not recognized.' % opt.dataset_mode)

    dataset.initialize(opt, split)
    print('Dataset [%s] was created (size: %d).' % (dataset.name(), len(dataset)))

    return dataset


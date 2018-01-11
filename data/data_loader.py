from __future__ import division, print_function

import torch.utils.data
import copy

# Todo: disentangle data-related parameters from model options

def CreateDataLoader(opt, is_train = None):
    # loader = CustomDataLoader()
    # loader.initialize(opt)
    # return loader

    if is_train is not None:
        # this is used when creating a val dataset (using test split and settings) during training
        # will be deprecated when data-related parameters are disentangled from model options
        opt = copy.deepcopy(opt)
        opt.is_train = is_train

    dataset = CreateDataset(opt)
    shuffle = (opt.shuffle==1)
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset, 
        batch_size = opt.batch_size,
        shuffle = shuffle, 
        num_workers = int(opt.nThreads), 
        drop_last = opt.is_train,
        pin_memory = True)

    return dataloader


def CreateDataset(opt):
    dataset = None

    if opt.dataset_mode == 'attribute':
        from data.attribute_dataset import AttributeDataset
        dataset = AttributeDataset()
    else:
        raise ValueError('Dataset mode [%s] not recognized.' % opt.dataset_mode)

    dataset.initialize(opt)
    print('Dataset [%s] was created (size: %d).' % (dataset.name(), len(dataset)))

    return dataset

class CustomDataLoader():

    def name(self):
        return 'CustomDataLoader'

    def initialize(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)

        shuffle = (opt.shuffle == 1)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = opt.batch_size,
            shuffle = shuffle,
            num_workers = int(opt.nThreads),
            drop_last = opt.is_train, # only drop last in training
            pin_memory = True)

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break

            yield data

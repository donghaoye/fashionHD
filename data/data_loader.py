from __future__ import division, print_function

import torch.utils.data

def CreateDataLoader(opt):
    loader = CustomDataLoader()
    loader.initialize(opt)
    return loader


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

        if opt.shuffle == 1:
            shuffle = True
        elif opt.shuffle == -1:
            shuffle = False
        else:
            # shuffle for training, not shuffle otherwise
            shuffle = opt.is_train

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

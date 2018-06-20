#-*- coding:utf-8 -*-
import torch.utils.data



def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset(opt)
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset(opt)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    return dataset


class CustomDatasetDataLoader(object):
    '''wrap the orianal torch.utils.data.DataLoader.
    '''
    def __init__(self, opt, dataset):
        '''
        args:
            opt: the opt object
            dataset: the data set object.
        '''
        self.opt = opt
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def name(self):
        return 'CustomDatasetDataLoader'

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data

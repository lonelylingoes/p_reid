#-*- coding:utf-8 -*-


def create_model(opt):
    model = None
    if opt.model == 'spgan':
        assert(opt.dataset_mode == 'unaligned')
        from .spgan_model import SPGANModel
        model = SPGANModel(opt)
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel(opt)
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    print("model [%s] was created" % (model.name()))
    return model

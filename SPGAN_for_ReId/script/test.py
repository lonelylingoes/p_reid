#-*- coding:utf-8 -*-
import os
from options.test_options import TestOptions
from data import CreateDataset
from data import CustomDatasetDataLoader
from models import create_model
from util.visualizer import save_images
from util import html


def main():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    dataset = CreateDataset(opt)
    data_loader = CustomDatasetDataLoader(opt, dataset)
    model = create_model(opt)
    # load the model's checkpoint; print networks; create shedulars
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir,  'testing_%s' % opt.which_epoch)
    webpage = html.HTML(web_dir, 'Testing, Experiment = %s,  Epoch = %s' % (opt.name, opt.which_epoch))
    # test
    for i, data in enumerate(data_loader):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_img_path()
        if i % opt.display_freq == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()


if __name__ == '__main__':
    main()
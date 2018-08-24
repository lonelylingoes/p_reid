from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()
        self.parser.add_argument('--display_freq', type=int, default=40, help='frequency of showing testing results on screen')
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='../results/market_to_duke', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.isTrain = False

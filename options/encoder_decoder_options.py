from multimodal_gan_options import BaseMMGANOptions

class EncoderDecoderOptions(BaseMMGANOptions):
    def initialize(self):
        super(EncoderDecoderOptions, self).initialize()
        parser = self.parser

        parser.add_argument('--decode_guided', action = 'store_true', help='use shape to guide decode')


    def auto_set(self):
        super(EncoderDecoderOptions, self).auto_set()
        opt = self.opt
        ###########################################
        # Add id profix
        if not opt.id.startswith('ED_'):
            opt.id = 'ED_' + opt.id
        ###########################################
        opt.edge_threshold = 0
        ###########################################

class TrainEncoderDecoderOptions(EncoderDecoderOptions):
    def initialize(self):
        super(TrainEncoderDecoderOptions, self).initialize()
        parser = self.parser
        # basic
        parser.add_argument('--continue_train', action = 'store_true', default = False, help = 'coninue training from saved model')
        
        # optimizer (we use Adam)
        parser.add_argument('--lr', type = float, default = 1e-4, help = 'initial learning rate')
        parser.add_argument('--beta1', type = float, default = 0.9, help = 'momentum1 term for Adam')
        parser.add_argument('--beta2', type = float, default = 0.999, help = 'momentum2 term for Adam')
        # scheduler
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau',
            choices = ['step', 'plateau', 'lambda'])
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--niter', type = int, default=15, help = '# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_decay', type=int, default=5, help='multiply by a gamma every lr_decay_interval epochs')
        parser.add_argument('--lr_gamma', type = float, default = 0.1, help='lr decay rate')
        parser.add_argument('--display_freq', type = int, default = 10, help='frequency of showing training results on screen')
        # parser.add_argument('--test_epoch_freq', type = int, default = 1, help='frequency of testing model')
        parser.add_argument('--save_epoch_freq', type = int, default = 5, help='frequency of saving model to disk' )
        parser.add_argument('--vis_epoch_freq', type = int, default = 1, help='frequency of visualizing generated images')
        parser.add_argument('--test_epoch_freq', type = int, default = 1, help='frequency of testing model')
        parser.add_argument('--max_n_vis', type = int, default = 32, help='max number of visualized images')

        # set train
        self.is_train = True

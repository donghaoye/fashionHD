from base_options import BaseOptions

class BaseAttributeOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)
        parser = self.parser

        # basic options
        parser.add_argument('--n_attr', type = int, default = 1000, help = 'number of attribute entries')
        parser.add_argument('--input_nc', type = int, default = 3, help = 'channel number of input images')
        parser.add_argument('--spatial_pool', type = str, default = 'none', help = 'spatial pooling method [max|noisy-or]',
            choices = ['max', 'noisyor', 'none'])
        parser.add_argument('--convnet', type = str, default = 'resnet101', help = 'CNN architecture [resnetX]')
        
        # data files
        # refer to "scripts/preproc_inshop.py" for more information
        parser.add_argument('--benchmark', type = str, default = 'ca', help = 'set benchmark [ca|inshop|user]',
            choices = ['ca', 'inshop', 'debug', 'user'])
        parser.add_argument('--fn_sample', type = str, default = 'default', help = 'path of sample index file')
        parser.add_argument('--fn_label', type = str, default = 'default', help = 'path of attribute label file')
        parser.add_argument('--fn_entry', type = str, default = 'default', help = 'path of attribute entry file')
        parser.add_argument('--fn_split', type = str, default = 'default', help = 'path of split file')

    def auto_set(self):
        super(BaseAttributeOptions, self).auto_set()

        if not self.opt.id.startswith('AE_'):
            self.opt.id = 'AE_' + self.opt.id

        if self.opt.benchmark == 'ca':
            if self.opt.fn_sample == 'default':
                self.opt.fn_sample = 'Label/ca_samples.json'
            if self.opt.fn_label == 'default':
                self.opt.fn_label = 'Label/ca_attr_label.pkl'
            if self.opt.fn_entry == 'default':
                self.opt.fn_entry = 'Label/attr_entry.json'
            if self.opt.fn_split == 'default':
                self.opt.fn_split = 'Split/ca_split_trainval.json'

        elif self.opt.benchmark == 'inshop':
            if self.opt.fn_sample == 'default':
                self.opt.fn_sample = 'Label/inshop_samples.json'
            if self.opt.fn_label == 'default':
                self.opt.fn_label = 'Label/inshop_attr_label.pkl'
            if self.opt.fn_entry == 'default':
                self.opt.fn_entry = 'Label/attr_entry.json'
            if self.opt.fn_split == 'default':
                self.opt.fn_split = 'Split/inshop_split.json'

        elif self.opt.benchmark == 'debug':
            self.opt.fn_sample = 'Label/debugca_samples.json'
            self.opt.fn_label = 'Label/debugca_attr_label.pkl'
            self.opt.fn_entry = 'Label/attr_entry.json'
            self.opt.fn_split = 'Split/debugca_split.json'





class TrainAttributeOptions(BaseAttributeOptions):

    def initialize(self):

        BaseAttributeOptions.initialize(self)
        parser = self.parser
        
        # train
        parser.add_argument('--continue_train', action = 'store_true', default = False, help = 'coninue training from saved model')
        parser.add_argument('--which_epoch', type = str, default = 'latest', help = 'which epoch to load? set to latest to use the latest cached model')
        parser.add_argument('--balanced', default = False, action = 'store_true', help = 'balanced loss weight for positive and negative samples')

        # optimizer (we use Adam)
        parser.add_argument('--optim', type = str, default = 'adam', help = 'optimizer type [adam|sgd]', 
            choices = ['adam', 'sgd'])
        parser.add_argument('--lr', type = float, default = 1e-4, help = 'initial learning rate')
        parser.add_argument('--beta1', type = float, default = 0.9, help = 'momentum term for Adam')

        # scheduler
        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau',
            choices = ['step', 'plateau', 'lambda'])
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--niter', type = int, default = 15, help = '# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr_decay', type=int, default=5, help='multiply by a gamma every lr_decay_interval epochs')
        self.parser.add_argument('--lr_gamma', type = float, default = 0.1, help='lr decay rate')

        self.parser.add_argument('--display_freq', type = int, default = 10, help='frequency of showing training results on screen')
        self.parser.add_argument('--test_epoch_freq', type = int, default = 1, help='frequency of testing model')
        self.parser.add_argument('--save_epoch_freq', type = int, default = 1, help='frequency of saving model to disk' )

        # misc
        self.parser.add_argument('--pavi', default = False, action = 'store_true', help = 'activate pavi log')


        # set train
        self.is_train = True

class TestAttributeOptions(BaseAttributeOptions):

    def initialize(self):

        BaseAttributeOptions.initialize(self)

        # test

        # set test
        self.is_train = False


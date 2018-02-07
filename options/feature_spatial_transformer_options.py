from base_options import BaseOptions

class BaseFeatureSpatialTransformerOptions(BaseOptions):

    def initialize(self):
        super(BaseFeatureSpatialTransformerOptions, self).initialize()
        parser = self.parser

        parser.add_argument('--which_model_AE', type = str, default = 'AE_2.6', help = 'pretrained attribute encoder ID')
        parser.add_argument('--shape_nc', type = int, default = 19, help = 'shape representation channel number')
        parser.add_argument('--feat_nc', type = int, default = 512, help = 'feature dimenssion')
        parser.add_argument('--shape_nf', type = int, default = 3, help = 'shape feat dimenssion of first conv layer')
        parser.add_argument('--n_shape_downsample', type = int, default = 5, help='downsample 5 times to reduce shape map from 22r*224 to 7*7')
        parser.add_argument('--reduce_type', type = str, default = 'pool', help = 'how to reduce a feature map to a non-spatial feature',
            choices = ['conv', 'pool'])
        parser.add_argument('--shape_encode', type = str, default = 'lm+seg', help = 'cloth shape encoding method',
            choices = ['lm', 'seg', 'lm+seg'])
        parser.add_argument('--input_mask_mode', type = str, default = 'map', help = 'type of segmentation mask. see base_dataset.segmap_to_mask for details. [foreground|body|target|map]',
            choices = ['foreground', 'body', 'target', 'map'])
        parser.add_argument('--benchmark', type = str, default = 'ca_upper', help = 'set benchmark [ca|ca_org|inshop|user|debug]',
            choices = ['ca', 'ca_upper', 'inshop', 'debug', 'user'])
        parser.add_argument('--fn_sample', type = str, default = 'default', help = 'path of sample index file')
        parser.add_argument('--fn_label', type = str, default = 'default', help = 'path of attribute label file')
        parser.add_argument('--fn_entry', type = str, default = 'default', help = 'path of attribute entry file')
        parser.add_argument('--fn_split', type = str, default = 'default', help = 'path of split file')
        parser.add_argument('--fn_landmark', type = str, default = 'default', help = 'path of landmark label file')
        parser.add_argument('--fn_seg_path', type = str, default = 'default', help = 'path of seg map list')
        # misc
        parser.add_argument('--batch_size', type = int, default = 128, help = 'batch size')
        self.parser.add_argument('--pavi', default = False, action = 'store_true', help = 'activate pavi log')


    def auto_set(self):
        super(BaseFeatureSpatialTransformerOptions, self).auto_set()
        opt = self.opt
        ###########################################
        # Add id profix
        if not opt.id.startswith('GAN_AE_'):
            opt.id = 'GAN_AE_' + opt.id

        nc_lm = 18
        nc_seg = 7 if opt.input_mask_mode == 'map' else 1

        if opt.shape_encode == 'lm':
            opt.shape_nc = nc_lm
        elif opt.shape_encode == 'seg':
            opt.shape_nc = nc_seg
        elif opt.shape_encode == 'lm+seg':
            opt.shape_nc = nc_lm + nc_seg

        ###########################################
        # Set default dataset file pathes
        if opt.benchmark.startswith('ca'):
            if opt.fn_sample == 'default':
                opt.fn_sample = 'Label/ca_samples.json'
            if opt.fn_label == 'default':
                opt.fn_label = 'Label/ca_attr_label.pkl'
            if opt.fn_entry == 'default':
                opt.fn_entry = 'Label/attr_entry.json'
            if opt.fn_landmark == 'default':
                opt.fn_landmark = 'Label/ca_landmark_label_256.pkl'
            if opt.fn_seg_path == 'default':
                opt.fn_seg_path = 'Label/ca_seg_paths.json'

            if opt.fn_split == 'default':
                if opt.benchmark == 'ca':
                    opt.fn_split = 'Split/ca_gan_split_trainval.json'
                elif opt.benchmark == 'ca_upper':
                    opt.fn_split = 'Split/ca_gan_split_trainval_upper.json'

        elif opt.benchmark == 'debug':
            opt.fn_sample = 'Label/debugca_gan_samples.json'
            opt.fn_label = 'Label/debugca_gan_attr_label.pkl'
            opt.fn_entry = 'Label/attr_entry.json'
            opt.fn_split = 'Split/debugca_gan_split.json'
            opt.fn_landmark = 'Label/debugca_gan_landmark_label.pkl'
            opt.fn_seg_path = 'Label/debugca_seg_paths.json'

        ###########################################

class TrainFeatureSpatialTransformerOptions(BaseFeatureSpatialTransformerOptions):
    def initialize(self):
        super(TrainFeatureSpatialTransformerOptions, self).initialize()
        parser = self.parser

        parser.add_argument('--continue_train', action = 'store_true', default = False, help = 'coninue training from saved model')
        # optimizer (we use Adam)
        parser.add_argument('--lr', type = float, default = 1e-4, help = 'initial learning rate')
        parser.add_argument('--weight_decay', type = float, default=5e-4, help='weight decay')
        parser.add_argument('--beta1', type = float, default = 0.9, help = 'momentum1 term for Adam')
        parser.add_argument('--beta2', type = float, default = 0.999, help = 'momentum2 term for Adam')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau',
            choices = ['step', 'plateau', 'lambda'])

        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--niter', type = int, default=15, help = '# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=5, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_decay', type=int, default=5, help='multiply by a gamma every lr_decay epochs')
        parser.add_argument('--lr_gamma', type = float, default = 0.1, help='lr decay rate')
        parser.add_argument('--display_freq', type = int, default = 10, help='frequency of showing training results on screen')
        parser.add_argument('--test_epoch_freq', type = int, default = 1, help='frequency of testing model')
        parser.add_argument('--save_epoch_freq', type = int, default = 1, help='frequency of saving model to disk' )

        # loss weights
        parser.add_argument('--loss_weight_L1', type = float, default = 1., help = 'loss weight of L1 loss')
        parser.add_argument('--loss_weight_attr', type = float, default = 0., help = 'loss weight of attribute BCE loss')

        # set train
        self.is_train = True


class TestFeatureSpatialTransformerOptions(BaseFeatureSpatialTransformerOptions):
    def initialize(self):
        self.is_train = False

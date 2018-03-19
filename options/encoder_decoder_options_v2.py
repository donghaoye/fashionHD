from base_options import BaseOptions

class BaseEncoderDecoderOptions_V2(BaseOptions):
    def initialize(self):
        super(BaseEncoderDecoderOptions_V2, self).initialize()
        parser = self.parser
        ##############################
        # General Setting
        ##############################
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [batch|instance|none]')
        parser.add_argument('--post_mask_mode', type = str, default = 'none', choices = ['none', 'fuse_face', 'fuse_face+bg'], help = 'how to mask generated images [none|fuse_face|fuse_face+bg]')
        parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
        parser.add_argument('--pavi', default = False, action = 'store_true', help = 'activate pavi log')
        ##############################
        # Encoder Setting
        ##############################
        parser.add_argument('--input_type', type=str, default='seg', choices = ['image', 'seg', 'edge'], help='type of encoder input')
        parser.add_argument('--nf', type=int, default=64)
        parser.add_argument('--nof', type=int, default=256)
        parser.add_argument('--max_nf', type=int, default=512)
        parser.add_argument('--ndowns', type=int, default=5)
        parser.add_argument('--nf_increase', type=str, default='exp', choices=['exp', 'linear'])
        parser.add_argument('--encode_fc', type=int, default=0, choices=[0,1])
        parser.add_argument('--feat_size', type=int, default=8)
        parser.add_argument('--block', type=str, default='residual', choices=['residual', 'conv'])
        ##############################
        # Decoder Setting
        ##############################
        parser.add_argument('--output_type', type=str, default='seg', choices = ['image', 'seg', 'edge'], help='type of decoder output and supervision')
        parser.add_argument('--decode_fc', type=int, default=0, choices=[0,1])
        parser.add_argument('--decode_guide', type=int, default=0, choices=[0,1])
        parser.add_argument('--gf', type=int, default=256, help='# channels of decode guidance feature map')
        ##############################
        # Auxiliary Encoder Setting
        ##############################
        parser.add_argument('--use_guide_encoder', type=int, default=0, choices=[0,1])
        parser.add_argument('--which_model_guide', type=str, default='EDV2_1.0.0')
        ##############################
        # data setting 1 fot gan dataset
        ##############################
        parser.add_argument('--affine_aug', type=int, default=0, choices=[0,1], help='apply random affine transformation on the input of encoders to disentangle desired information from shape')
        parser.add_argument('--affine_aug_scale', type=float, default=0.05, help='scale of random affine transformation augmentation')
        
        parser.add_argument('--shape_encode', type = str, default = 'seg', choices = ['lm', 'seg', 'lm+seg', 'seg+e', 'lm+seg+e', 'e', 'reduced_seg', 'flx_seg'], help = 'cloth shape encoding method')

        parser.add_argument('--edge_mode', type=str, default='cloth', choices=['outer', 'inner', 'cloth'], help='outer: all edges; inner: edge inside the clothing mask, no contour; cloth: edge inside the clothing mask, with contour')
        parser.add_argument('--edge_threshold', type=int, default=0, help='edge threshold to filter small edge [0-255]')

        parser.add_argument('--color_jitter', type=int, default=1, choices=[0,1], help='use color jitter augmentation')
        parser.add_argument('--color_gaussian_ksz', type=int, default=15, help='gaussian blur kernel size')
        parser.add_argument('--color_gaussian_sigma', type=float, default=10.0, help='gaussian blur sigma')
        parser.add_argument('--color_patch', type=int, default=1, choices=[0,1], help='use a patch inside the clothing region')
        parser.add_argument('--color_patch_mode', type=str, default='crop5', choices=['center', 'crop5', 'single'], help='method to extract patch')

        parser.add_argument('--benchmark', type = str, default = 'ca_upper', choices = ['ca', 'ca_upper', 'inshop', 'debug', 'user'], help = 'set benchmark [ca|ca_org|inshop|user|debug]')
        parser.add_argument('--fn_sample', type = str, default = 'default', help = 'path of sample index file')
        parser.add_argument('--fn_label', type = str, default = 'default', help = 'path of attribute label file')
        parser.add_argument('--fn_entry', type = str, default = 'default', help = 'path of attribute entry file')
        parser.add_argument('--fn_split', type = str, default = 'default', help = 'path of split file')
        parser.add_argument('--fn_landmark', type = str, default = 'default', help = 'path of landmark label file')
        parser.add_argument('--fn_seg_path', type = str, default = 'default', help = 'path of seg map')
        parser.add_argument('--fn_flx_seg_path', type = str, default = 'default', help = 'path of uncertain seg map')
        parser.add_argument('--fn_edge_path', type = str, default = 'default', help = 'path of edge map')
        parser.add_argument('--fn_color_path', type = str, default = 'default', help = 'path of color map')

    def auto_set(self):
        super(BaseEncoderDecoderOptions_V2, self).auto_set()
        opt = self.opt
        ###########################################
        # Add id profix
        ###########################################
        if not opt.id.startswith('EDV2_'):
            opt.id = 'EDV2_' + opt.id
        ###########################################
        # Check feat size
        ###########################################
        if opt.decode_fc:
            assert opt.encode_fc
            assert opt.feat_size == 1
        ###########################################
        # Check guide encoder
        ###########################################
        if opt.decode_guide:
            opt.use_guide_encoder = 1

        ###########################################
        # set dataset file path
        ###########################################
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
                # opt.fn_seg_path = 'Label/ca_seg_paths.json'
                opt.fn_seg_path = 'Label/ca_syn_seg_paths.json'
            if opt.fn_flx_seg_path == 'default':
                opt.fn_flx_seg_path = 'Label/ca_gan_flx_seg_paths.json'
            if opt.fn_edge_path == 'default':
                if opt.edge_mode == 'outer':
                    opt.fn_edge_path = 'Label/ca_edge_paths.json'
                elif opt.edge_mode == 'inner':
                    opt.fn_edge_path = 'Label/ca_edge_inner_paths.json'
                elif opt.edge_mode == 'cloth':
                    opt.fn_edge_path = 'Label/ca_edge_cloth_paths.json'
            if opt.fn_color_path == 'default':
                opt.fn_color_path = 'Label/ca_edge_paths.json'# Todo: modify this temp setting
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
            opt.fn_flx_seg_path = 'Label/debugca_gan_flx_seg_paths.json'
            opt.fn_edge_path = 'Label/debugca_edge_paths.json'
            opt.fn_color_path = 'Label/debugca_edge_paths.json' # Todo: modify this temp setting


class TrainEncoderDecoderOptions_V2(BaseEncoderDecoderOptions_V2):
    def initialize(self):
        super(TrainEncoderDecoderOptions_V2, self).initialize()
        parser = self.parser

        # basic
        parser.add_argument('--continue_train', action = 'store_true', default = False, help = 'coninue training from saved model')
        # optimizer
        parser.add_argument('--lr', type = float, default = 1e-4, help = 'initial learning rate')
        parser.add_argument('--beta1', type = float, default = 0.9, help = 'momentum1 term for Adam')
        parser.add_argument('--beta2', type = float, default = 0.999, help = 'momentum2 term for Adam')
        # scheduler
        parser.add_argument('--lr_policy', type=str, default='step', choices = ['step', 'plateau', 'lambda'], help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--niter', type = int, default=9, help = '# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_decay', type=int, default=3, help='multiply by a gamma every lr_decay_interval epochs')
        parser.add_argument('--lr_gamma', type = float, default = 0.1, help='lr decay rate')
        parser.add_argument('--display_freq', type = int, default = 10, help='frequency of showing training results on screen')
        parser.add_argument('--test_epoch_freq', type = int, default = 1, help='frequency of testing model')
        parser.add_argument('--save_epoch_freq', type = int, default = 5, help='frequency of saving model to disk' )
        parser.add_argument('--vis_epoch_freq', type = int, default = 1, help='frequency of visualizing generated images')
        parser.add_argument('--max_n_vis', type = int, default = 32, help='max number of visualized images')
        # loss weights
        parser.add_argument('--loss_weight_decode', type=float, default=1)

        # set train
        self.is_train = True

class TestEncoderDecoderOptions_V2(BaseEncoderDecoderOptions_V2):
    def initialize(self):
        super(TestEncoderDecoderOptions_V2, self).initialize()
        parser = self.parser
        # set test
        self.is_train = False

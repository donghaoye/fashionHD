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
        parser.add_argument('--input_type', type=str, default='shape', choices = ['image', 'seg', 'edge', 'shape'], help='type of encoder input')
        parser.add_argument('--nf', type=int, default=64)
        parser.add_argument('--nof', type=int, default=256)
        parser.add_argument('--max_nf', type=int, default=512)
        parser.add_argument('--ndowns', type=int, default=5)
        parser.add_argument('--nf_increase', type=str, default='exp', choices=['exp', 'linear'])
        parser.add_argument('--encode_fc', type=int, default=0, choices=[0,1])
        parser.add_argument('--feat_size', type=int, default=8)
        parser.add_argument('--block', type=str, default='conv', choices=['residual', 'conv'])
        ##############################
        # Decoder Setting
        ##############################
        parser.add_argument('--output_type', type=str, default='seg', choices = ['image', 'seg', 'edge'], help='type of decoder output and supervision')
        parser.add_argument('--decode_fc', type=int, default=0, choices=[0,1])
        parser.add_argument('--decode_guide', type=int, default=0, choices=[0,1])
        parser.add_argument('--decode_gf', type=int, default=256, help='# channels of decode guidance feature map')
        ##############################
        # Guide Encoder Setting
        ##############################
        parser.add_argument('--use_guide_encoder', type=int, default=0, choices=[0,1])
        parser.add_argument('--which_model_guide', type=str, default='EDV2_3.0')
        ##############################
        # DFN
        ##############################
        parser.add_argument('--use_dfn', type=int, default=1, choices=[0,1], help='set as 0 to disable DFN')
        parser.add_argument('--dfn_nmid', type=int, default=64, help='mid-level channel number of DFN')
        parser.add_argument('--dfn_local_size', type=int, default=5, help='local region size')
        parser.add_argument('--dfn_detach', type=int, default=0, choices=[0,1], help='detach output feature from input feature in DFN')
        parser.add_argument('--dfn_nblocks', type=int, default=2, help='number of resnet blocks following the DFN network')
        ##############################
        # GAN
        ##############################
        parser.add_argument('--which_gan', type=str, default='dcgan', choices=['dcgan', 'lsgan'], help='gan loss type')
        parser.add_argument('--gan_level', type=str, default='image', choices=['image', 'feature'], help='apply gan loss at image level or feature level')
        parser.add_argument('--D_cond', type=int, default=0, choices=[0,1], help='use conditioned discriminator')
        parser.add_argument('--D_cond_type', type=str, default='cond', choices=['cond', 'pair'], help='cond: input+output, pair: output+dual_output')
        ##############################
        # data setting (dataset_mode == gan_v2)
        ##############################
        parser.add_argument('--dataset_mode', type=str, default='gan_v2', help='type of dataset. see data/data_loader.py')
        parser.add_argument('--benchmark', type=str, default = 'zalando', help='which dataset [zalando|ca_upper]')
        parser.add_argument('--debug', action='store_true', help='debug')
        parser.add_argument('--data_root', type=str, default='./datasets/Zalando', help='data root path')
        parser.add_argument('--img_dir', type=str, default='Img/img_zalando_256/')
        parser.add_argument('--seg_dir', type=str, default='Img/seg_zalando_256/')
        parser.add_argument('--edge_dir', type=str, default='Img/edge_zalando_256_cloth/')
        parser.add_argument('--fn_split', type=str, default='Split/zalando_split.json')
        parser.add_argument('--fn_pose', type=str, default='Label/zalando_pose_label_256.pkl')

        parser.add_argument('--seg_bin_size', type=int, default=16, help='bin size of downsampled seg mask')
        parser.add_argument('--pose_size', type=int, default=11, help='point size in pose heatmap')
        parser.add_argument('--edge_threshold', type=int, default=0, help='edge score threshold')
        parser.add_argument('--color_gaussian_ksz', type=int, default=15)
        parser.add_argument('--color_gaussian_sigma', type=float, default=10.0)
        parser.add_argument('--color_bin_size', type=int, default=16)
        parser.add_argument('--color_jitter', type=int, default=1)
        parser.add_argument('--shape_deformation_scale', type=float, default=0.1)
        parser.add_argument('--shape_deformation_flip', type=int, default=1)

        
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
        
        


class TrainEncoderDecoderOptions_V2(BaseEncoderDecoderOptions_V2):
    def initialize(self):
        super(TrainEncoderDecoderOptions_V2, self).initialize()
        parser = self.parser

        # basic
        parser.add_argument('--continue_train', action = 'store_true', default = False, help = 'coninue training from saved model')
        # optimizer
        parser.add_argument('--lr', type = float, default = 1e-4, help = 'initial learning rate')
        parser.add_argument('--lr_D', type = float, default = 1e-5, help = 'only use lr_D for netD when loss_weight_gan > 0')
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
        parser.add_argument('--loss_weight_trans', type=float, default=1)
        parser.add_argument('--loss_weight_cycle', type=float, default=1)
        parser.add_argument('--loss_weight_gan', type=float, default=0., help='set loss_weight_gan > 0 to enable GAN loss')
        # set train
        self.is_train = True

class TestEncoderDecoderOptions_V2(BaseEncoderDecoderOptions_V2):
    def initialize(self):
        super(TestEncoderDecoderOptions_V2, self).initialize()
        parser = self.parser
        # set test
        self.is_train = False

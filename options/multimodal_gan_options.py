from base_options import BaseOptions

class BaseMMGANOptions(BaseOptions):

    def initialize(self):
        super(BaseMMGANOptions, self).initialize()
        parser = self.parser

        ##############################
        # netG and netD
        ##############################
        
        # will be set in auto_set()
        parser.add_argument('--G_input_nc', type = int, default = 19, help = '# of netG input channels. default value is 19 = 18(landmark heatmap) + 1(segmentation map)')
        parser.add_argument('--G_output_nc', type = int, default = 3, help = '# of netG output channels')
        parser.add_argument('--G_cond_nc', type=int, default=0, help='# of netG condition channels')
        parser.add_argument('--D_input_nc', type = int, default = 22, help = '# of netD input channels')

        parser.add_argument('--G_cond_size', type = int, default=56, help='224x224 image will have 56x56 feature map')
        parser.add_argument('--D_no_cond', action='store_true', help='only input images(no condition info) into netD')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--which_model_netG', type = str, default = 'resnet_6blocks', help = 'select model to use for netG')
        parser.add_argument('--which_model_netD', type = str, default = 'basic', help = 'select model to use for netD')
        parser.add_argument('--which_model_init', type = str, default = 'none', help = 'load pretrained model to init netG parameters')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [batch|instance|none]')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--G_cond_layer', type = str, default = 'first', help = 'which layer to add condition feature',
            choices = ['first', 'all'])
        parser.add_argument('--G_cond_interp', type = str, default = 'bilinear', help = 'interpolation when upsample condition feature map to desired size',
            choices = ['bilinear', 'nearest'])
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--which_gan', type=str, default='dcgan', help='type of gan loss [dcgan|lsgan|wgan]',
            choices = ['dcgan', 'lsgan', 'wgan'])
        parser.add_argument('--shape_encode', type = str, default = 'seg', help = 'cloth shape encoding method',
            choices = ['lm', 'seg', 'lm+seg', 'seg+e', 'lm+seg+e', 'e'])
        parser.add_argument('--shape_nc', type=int, default=0, help='# channels of shape representation, depends on shape_emcode, will be auto set')
        parser.add_argument('--input_mask_mode', type = str, default = 'map', help = 'type of segmentation mask. see base_dataset.segmap_to_mask for details. [foreground|body|target|map]',
            choices = ['foreground', 'body', 'target', 'map', 'grid_map'])
        parser.add_argument('--post_mask_mode', type = str, default = 'fuse_face', help = 'how to mask generated images [none|fuse_face|fuse_face+bg]',
            choices = ['none', 'fuse_face', 'fuse_face+bg'])
        # none: do not mask image
        # fuse_face: replace the face&hair part with original image
        # fuse_face_bg: replace the face&hair&background part with original image

        ##############################
        # encoder general setting
        ##############################
        parser.add_argument('--affine_aug', action='store_true', help='apply random affine transformation on the input of encoders to disentangle desired information from shape')
        parser.add_argument('--affine_aug_scale', type=float, default=0.05, help='scale of random affine transformation augmentation')
        parser.add_argument('--encoder_type', type=str, default='normal', help='network architecture of encoder',
            choices = ['normal', 'pool', 'fc', 'st'])
        parser.add_argument('--encoder_attention', type=int, default=0, help='only useful when encoder_type is "pool". set 1 to use attention weighted pooling',
            choices = [0,1])
        parser.add_argument('--encoder_block', type=str, default='residual', help='block type of downsample layers in encoder networks',
            choices = ['residual', 'downsample'])
        parser.add_argument('--color_jitter', action='store_true', default='use color jitter augmentation')
        parser.add_argument('--tar_guided', type=int, default=1, help='use target shape to guide encoding. available for STImageEncoder')
        # parser.add_argument('--encoder_norm', type=str, default='instance', help='norm layers in encoder net',
        #     choices = ['instance', 'batch'])
        ##############################
        # attribute encoder
        ##############################
        parser.add_argument('--use_attr', action='store_true', help='use attribute encoder')
        parser.add_argument('--which_model_AE', type = str, default = 'AE_2.6', help = 'pretrained attribute encoder ID')
        parser.add_argument('--n_attr', type = int, default = 1000, help = 'number of attribute entries')
        parser.add_argument('--n_attr_feat', type = int, default = 512, help = '# of attribute feature channels')
        parser.add_argument('--attr_cond_type', type = str, default = 'feat_map', help = 'attribute condition form [feat|prob|none]',
            choices = ['feat', 'prob', 'feat_map', 'prob_map'])
        ##############################
        # edge encoder
        ##############################
        parser.add_argument('--use_edge', action = 'store_true', help='use edge condition branch')
        parser.add_argument('--edge_outer', action = 'store_true', help='use all edges instead of inner edge')
        parser.add_argument('--edge_threshold', type=int, default=0, help='edge threshold to filter small edge [0-255]')
        parser.add_argument('--edge_nf', type=int, default=64, help='feature dimension of first conv layer in edge encoder')
        parser.add_argument('--edge_nof', type=int, default=128, help='output feature dimension, set -1  to use default setting')
        parser.add_argument('--edge_ndowns',type=int, default=5, help='number of downsample layers in edge encoder')
        parser.add_argument('--edge_shape_guided', type=int, default=0, choices=[0,1], help='concat shape_mask and edge_map to guide edge encoding')
        parser.add_argument('--edge_encoder_type', type=str, default='default')
        parser.add_argument('--edge_encoder_block', type=str, default='default')
        ##############################
        # color encoder
        ##############################
        parser.add_argument('--use_color', action='store_true', help='use color condition branch')
        parser.add_argument('--color_nf', type=int, default=64, help='feature dimension of first conv layer in color encoder')
        parser.add_argument('--color_nof', type=int, default=128, help='output feature dimension, set -1  to use default setting')
        parser.add_argument('--color_ndowns', type=int, default=5, help='number of downsample layers in color encoder')
        parser.add_argument('--color_shape_guided', type=int, default=0, choices=[0,1], help='concat shape_mask and color_map to guide color encoding')
        parser.add_argument('--color_gaussian_ksz', type=int, default=15, help='gaussian blur kernel size')
        parser.add_argument('--color_gaussian_sigma', type=float, default=10.0, help='gaussian blur sigma')
        parser.add_argument('--color_patch', action='store_true', help='use a patch inside the clothing region')
        parser.add_argument('--color_patch_mode', type=str, default='center', choices=['center', 'crop5', 'single'], help='method to extract patch')
        parser.add_argument('--color_encoder_type', type=str, default='default')
        parser.add_argument('--color_encoder_block', type=str, default='default')
        ##############################
        # data (refer to "scripts/preproc_inshop.py" for more information)
        ##############################
        parser.add_argument('--benchmark', type = str, default = 'ca_upper', help = 'set benchmark [ca|ca_org|inshop|user|debug]',
            choices = ['ca', 'ca_upper', 'inshop', 'debug', 'user'])
        parser.add_argument('--fn_sample', type = str, default = 'default', help = 'path of sample index file')
        parser.add_argument('--fn_label', type = str, default = 'default', help = 'path of attribute label file')
        parser.add_argument('--fn_entry', type = str, default = 'default', help = 'path of attribute entry file')
        parser.add_argument('--fn_split', type = str, default = 'default', help = 'path of split file')
        parser.add_argument('--fn_landmark', type = str, default = 'default', help = 'path of landmark label file')
        parser.add_argument('--fn_seg_path', type = str, default = 'default', help = 'path of seg map')
        parser.add_argument('--fn_edge_path', type = str, default = 'default', help = 'path of edge map')
        parser.add_argument('--fn_color_path', type = str, default = 'default', help = 'path of color map')
        parser.add_argument('--fn_flx_seg_path', type = str, default = 'default', help = 'path of uncertain seg map')
        ##############################
        # auxiliary discriminators
        ##############################
        parser.add_argument('--use_edge_D', action='store_true', help='use auxiliary edge matching discriminator')
        parser.add_argument('--use_color_D', action='store_true', help='use auxiliary color matching discriminator')
        parser.add_argument('--use_attr_D', action='store_true', help='use auxiliary attribute matching discriminator')
        ##############################
        # misc
        ##############################
        parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
        self.parser.add_argument('--pavi', default = False, action = 'store_true', help = 'activate pavi log')


    def auto_set(self):
        super(BaseMMGANOptions, self).auto_set()
        opt = self.opt
        ###########################################
        # Add id profix
        if not opt.id.startswith('MMDGAN_RECON_'):
            opt.id = 'MMDGAN_RECON_' + opt.id
        ###########################################
        # set dimmension settings
        nc_img = 3
        nc_lm = 18
        nc_edge = 1
        nc_color = 6 if (opt.color_patch and opt.color_patch_mode != 'single') else 3
        if opt.input_mask_mode == 'map':
            nc_seg = 7
        elif opt.input_mask_mode == 'grid_map':
            nc_seg = 9
        else:
            nc_seg = 1
        nf_edge = min(512, opt.edge_nf * 2**(opt.edge_ndowns)) if opt.edge_nof == -1 else opt.edge_nof
        nf_color = min(512, opt.color_nf * 2**(opt.color_ndowns)) if opt.color_nof == -1 else opt.color_nof
        nf_attr = opt.n_attr_feat if opt.attr_cond_type in {'feat', 'feat_map'} else opt.n_attr

        # set netG input_nc and output_nc
        if opt.shape_encode == 'lm':
            opt.shape_nc = nc_lm
        elif opt.shape_encode == 'seg':
            opt.shape_nc = nc_seg
        elif opt.shape_encode == 'lm+seg':
            opt.shape_nc = nc_lm + nc_seg
        elif opt.shape_encode == 'lm+seg+e':
            opt.shape_nc = nc_lm + nc_seg + nc_edge
        elif opt.shape_encode == 'seg+e':
            opt.shape_nc = nc_seg + nc_edge
        elif opt.shape_encode == 'e':
            opt.shape_nc = nc_edge
        
        opt.G_input_nc = opt.shape_nc
        opt.G_output_nc = nc_img

        # set netG cond_nc, netD input_nc
        opt.G_cond_nc = 0
        opt.D_input_nc = nc_img + opt.G_input_nc
        if opt.use_edge:
            opt.G_cond_nc += nf_edge
            opt.D_input_nc += nc_edge
        if opt.use_color:
            opt.G_cond_nc += nf_color
            opt.D_input_nc += nc_color
        if opt.use_attr:
            opt.G_cond_nc += nf_attr
            # don not input attribute conditions into netD
        if opt.D_no_cond:
            opt.D_input_nc = nc_img

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
                # opt.fn_seg_path = 'Label/ca_seg_paths.json'
                opt.fn_seg_path = 'Label/ca_syn_seg_paths.json'
            if opt.fn_flx_seg_path == 'default':
                opt.fn_flx_seg_path = 'Label/ca_gan_flx_seg_paths.json'
            if opt.fn_edge_path == 'default':
                if opt.edge_outer:
                    opt.fn_edge_path = 'Label/ca_edge_paths.json'
                else:
                    opt.fn_edge_path = 'Label/ca_edge_inner_paths.json'
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

        ###########################################

        
            
class TrainMMGANOptions(BaseMMGANOptions):

    def initialize(self):

        super(TrainMMGANOptions,self).initialize()
        parser = self.parser
        
        # basic
        parser.add_argument('--continue_train', action = 'store_true', default = False, help = 'coninue training from saved model')

        # train setting
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        
        # optimizer (we use Adam)
        parser.add_argument('--lr', type = float, default = 2e-4, help = 'initial learning rate')
        parser.add_argument('--lr_D', type=float, default=2e-4, help='initial learning rate for netD')
        parser.add_argument('--beta1', type = float, default = 0.5, help = 'momentum1 term for Adam')
        parser.add_argument('--beta2', type = float, default = 0.999, help = 'momentum2 term for Adam')
        # scheduler
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau',
            choices = ['step', 'plateau', 'lambda'])
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--niter', type = int, default=20, help = '# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=5, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_decay', type=int, default=1, help='multiply by a gamma every lr_decay_interval epochs')
        parser.add_argument('--lr_gamma', type = float, default = 0.1, help='lr decay rate')
        parser.add_argument('--display_freq', type = int, default = 10, help='frequency of showing training results on screen')
        # parser.add_argument('--test_epoch_freq', type = int, default = 1, help='frequency of testing model')
        parser.add_argument('--save_epoch_freq', type = int, default = 5, help='frequency of saving model to disk' )
        parser.add_argument('--vis_epoch_freq', type = int, default = 1, help='frequency of visualizing generated images')
        parser.add_argument('--max_n_vis', type = int, default = 32, help='max number of visualized images')
        parser.add_argument('--D_pretrain', type = int, default = 0, help = 'iter num of pretraining net D')
        parser.add_argument('--D_train_freq', type = int, default = 1, help='frequency of training netD')
        parser.add_argument('--G_train_freq', type = int, default = 1, help='frequency of training netG')
        parser.add_argument('--check_grad_freq', type = int, default = 100, help = 'frequency of checking gradient of each loss')

        # loss weights
        parser.add_argument('--loss_weight_GAN', type = float, default = 1., help = 'loss wweight of GAN loss (for netG)')
        parser.add_argument('--loss_weight_L1', type = float, default = 100., help = 'loss weight of L1 loss')
        parser.add_argument('--loss_weight_vgg', type = float, default = 100., help = 'loss weight of vgg loss (perceptual feature loss)')
        parser.add_argument('--loss_weight_gp', type = float, default = 10., help = 'gradient penalty weight in WGAN')

        # training method
        # parser.add_argument('--shape_adaptive', action='store_true', help='add training samples with unmatched shape/attribute representation, \
        #     and optimize only GAN loss for these samples to force netG to generate realistic images from unmatched conditions')
        
        # joint train modules

        # set train
        self.is_train = True

class TestMMGANOptions(BaseMMGANOptions):

    def initialize(self):

        super(TestMMGANOptions, self).initialize()
        parser = self.parser
        
        # test

        # set test
        self.is_train = False


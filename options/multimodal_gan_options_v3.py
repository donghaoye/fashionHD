from base_options import BaseOptions

class BaseMMGANOptions_V3(BaseOptions):

    def initialize(self):
        super(BaseMMGANOptions_V3, self).initialize()
        parser = self.parser
        ##############################
        # General Setting
        ##############################
        parser.add_argument('--which_gan', type=str, default='dcgan', choices = ['dcgan', 'lsgan', 'wgan'], help='type of gan loss [dcgan|lsgan|wgan]')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [batch|instance|none]')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--input_mask_mode', type = str, default = 'map', choices = ['foreground', 'body', 'target', 'map', 'grid_map'], help = 'type of segmentation mask. see base_dataset.segmap_to_mask for details. [foreground|body|target|map]')
        parser.add_argument('--post_mask_mode', type = str, default = 'none', choices = ['none', 'fuse_face', 'fuse_face+bg'], help = 'how to mask generated images [none|fuse_face|fuse_face+bg]')
        parser.add_argument('--batch_size', type = int, default = 16, help = 'batch size')
        parser.add_argument('--pavi', default = False, action = 'store_true', help = 'activate pavi log')
        parser.add_argument('--which_model_init', type = str, default = 'none', help = 'load pretrained model to init netG parameters')
        ##############################
        # Encoder General Setting
        ##############################
        parser.add_argument('--affine_aug', action='store_true', help='apply random affine transformation on the input of encoders to disentangle desired information from shape')
        parser.add_argument('--affine_aug_scale', type=float, default=0.05, help='scale of random affine transformation augmentation')
        parser.add_argument('--color_jitter', type=int, default=1, choices=[0,1], help='use color jitter augmentation')
        parser.add_argument('--encoder_type', type=str, default='normal', choices = ['normal', 'pool', 'fc', 'st'], help='network architecture of encoder')
        parser.add_argument('--encoder_block', type=str, default='residual', choices = ['residual', 'downsample'], help='block type of downsample layers in encoder networks')
        parser.add_argument('--tar_guided', type=int, default=1, help='use target shape to guide encoding. available for STImageEncoder')
        parser.add_argument('--encoder_attention', type=int, default=0, choices = [0,1], help='only useful when encoder_type is "pool". set 1 to use attention weighted pooling')
        ##############################
        # Shape Encoder
        ##############################
        parser.add_argument('--shape_nf', type=int, default=64, help='feature dimension of first conv layer in shape encoder')
        parser.add_argument('--shape_nof', type=int, default=128, help='output feature dimension, set -1  to use default setting')
        parser.add_argument('--shape_ndowns',type=int, default=5, help='number of downsample layers in shape encoder')
        parser.add_argument('--shape_encoder_type', type=str, default='default')
        parser.add_argument('--shape_encoder_block', type=str, default='default')
        parser.add_argument('--pretrain_shape', type=int, default=1, choices=[0,1], help='load pretrained shape_encoder')
        parser.add_argument('--which_model_init_shape_encoder', type=str, default='default', help='id of pretrained shape encoder')
        parser.add_argument('--shape_encode', type = str, default = 'reduced_seg', choices = ['lm', 'seg', 'lm+seg', 'seg+e', 'lm+seg+e', 'e', 'reduced_seg', 'flx_seg'], help = 'cloth shape encoding method')
        parser.add_argument('--shape_with_face', type= int, default = 1, choices = [0,1], help='add face region rgb information into shape representation')
        parser.add_argument('--shape_nc', type=int, default=0, help='# channels of shape representation, depends on shape_emcode, will be auto set')
        ##############################
        # Edge Encoder
        ##############################
        parser.add_argument('--use_edge', type=int, default=1, choices=[0,1], help='use edge condition branch')
        parser.add_argument('--edge_nf', type=int, default=64, help='feature dimension of first conv layer in edge encoder')
        parser.add_argument('--edge_nof', type=int, default=128, help='output feature dimension, set -1  to use default setting')
        parser.add_argument('--edge_ndowns',type=int, default=5, help='number of downsample layers in edge encoder')
        parser.add_argument('--edge_shape_guided', type=int, default=0, choices=[0,1], help='concat shape_mask and edge_map to guide edge encoding')
        # parser.add_argument('--edge_outer', type=int, default=1, choices=[0, 1], help='use all edges instead of inner edge')
        parser.add_argument('--edge_mode', type=str, default='cloth', choices=['outer', 'inner', 'cloth'], help='outer: all edges; inner: edge inside the clothing mask, no contour; cloth: edge inside the clothing mask, with contour')
        parser.add_argument('--edge_threshold', type=int, default=0, help='edge threshold to filter small edge [0-255]')
        parser.add_argument('--edge_encoder_type', type=str, default='default')
        parser.add_argument('--edge_encoder_block', type=str, default='default')
        parser.add_argument('--pretrain_edge', type=int, default=1, choices=[0,1], help='load pretrained edge encoder')
        parser.add_argument('--which_model_init_edge_encoder', type=str, default='default', help='id of pretrained edge encoder')
        ##############################
        # Color Encoder
        ##############################
        parser.add_argument('--use_color', type=int, default=1, choices=[0,1], help='use color condition branch')
        parser.add_argument('--color_nf', type=int, default=64, help='feature dimension of first conv layer in color encoder')
        parser.add_argument('--color_nof', type=int, default=128, help='output feature dimension, set -1  to use default setting')
        parser.add_argument('--color_ndowns', type=int, default=5, help='number of downsample layers in color encoder')
        parser.add_argument('--color_shape_guided', type=int, default=0, choices=[0,1], help='concat shape_mask and color_map to guide color encoding')
        parser.add_argument('--color_gaussian_ksz', type=int, default=15, help='gaussian blur kernel size')
        parser.add_argument('--color_gaussian_sigma', type=float, default=10.0, help='gaussian blur sigma')
        parser.add_argument('--color_patch', type=int, default=1, choices=[0,1], help='use a patch inside the clothing region')
        parser.add_argument('--color_patch_mode', type=str, default='crop5', choices=['center', 'crop5', 'single'], help='method to extract patch')
        parser.add_argument('--color_encoder_type', type=str, default='default')
        parser.add_argument('--color_encoder_block', type=str, default='default')
        parser.add_argument('--pretrain_color', type=int, default=1, choices=[0,1], help='load pretrained color encoder')
        parser.add_argument('--which_model_init_color_encoder', type=str, default='default', help='id of pretrained color encoder')
        ##############################
        # Feature Transfer and Fusion
        ##############################
        parser.add_argument('--feat_size_lr', type=int, default=8, help='LR feature map size (input size of netG)')
        parser.add_argument('--feat_size_hr', type=int, default=64, help='HR feature map size (input size of shape guide)')
        ##############################
        # Generator
        ##############################
        parser.add_argument('--which_model_netG', type = str, default = 'unet', choices = ['decoder', 'unet'], help='select model to use for netG')
        parser.add_argument('--G_output_seg', type=int, default=1, choices=[0,1], help='generator output image and (7-channel) segentation map')
        parser.add_argument('--G_output_nc', type=int, default=3, help='# output channels of netG')
        # for decoder generator
        parser.add_argument('--G_shape_guided', action='store_true', help='add shape guide at LR level')
        parser.add_argument('--G_nblocks_1', type=int, default=2, help='number of LR residual blocks')
        parser.add_argument('--G_nups_1', type=int, default=3, help='number of LR upsample layers')
        parser.add_argument('--G_nblocks_2', type=int, default=6, help='number of HR residual blocks')
        parser.add_argument('--G_nups_2', type=int, default=2, help='number of HR upsample layers')
        # for unet generator
        parser.add_argument('--G_ndowns', type=int, default=5, help='number of downsample layers')
        parser.add_argument('--G_nblocks', type=int, default=3, help='number of residual block')
        parser.add_argument('--G_block', type=str, default='normal',choices = ['normal', 'residual'], help='generator block type')
        ##############################
        # Discriminator
        ##############################
        parser.add_argument('--which_model_netD', type = str, default = 'basic', help = 'select model to use for netD')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--D_input_nc', type = int, default = 22, help = '# of netD input channels')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--D_no_cond', action='store_true', help='only input images(no condition info) into netD')
        ##############################
        # data setting 1 fot gan dataset
        ##############################
        parser.add_argument('--benchmark', type = str, default = 'ca_upper', choices = ['ca', 'ca_upper', 'inshop', 'debug', 'user'], help = 'set benchmark [ca|ca_org|inshop|user|debug]')
        parser.add_argument('--fn_split', type = str, default = 'Split/ca_gan_split_trainval_upper.json', help = 'path of split file')
        parser.add_argument('--fn_landmark', type = str, default = 'Label/ca_landmark_label_256.pkl', help = 'path of landmark label file')
        ##############################
        # data setting 2 for aligned dataset
        ##############################
        parser.add_argument('--debug', action='store_true', help='use debug split of dataset (32 samples)')
        parser.add_argument('--img_dir', type=str, default='Img/img_ca_256/')
        parser.add_argument('--seg_dir', type=str, default='Img/seg_ca_syn_256/')
        parser.add_argument('--flx_seg_dir', type=str, default='Img/seg_ca_syn_256_flexible/')
        parser.add_argument('--edge_dir', type=str, default='Img/edge_ca_256_cloth/')
        parser.add_argument('--edge_warp_dir', type=str, default='Img/edge_ca_256_cloth_tps/')

    def auto_set(self):
        super(BaseMMGANOptions_V3, self).auto_set()
        opt = self.opt
        ###########################################
        # Add id profix
        ###########################################
        if not opt.id.startswith('V3_'):
            opt.id = 'V3_' + opt.id
        ###########################################
        # set for aligned dataset
        ###########################################
        opt.dataset_mode = 'aligned_gan'
        ###########################################
        # set dimmension settings
        ###########################################
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
        # set shape_encoder
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
        elif opt.shape_encode == 'reduced_seg':
            opt.shape_nc = 4
        elif opt.shape_encode == 'flx_seg':
            opt.shape_nc = 7
        if opt.shape_with_face:
            opt.shape_nc += 3
        # set G_output_nc
        if opt.G_output_seg:
            opt.G_output_nc = 10
        else:
            opt.G_output_nc = 3
        # set D_input_nc
        if opt.D_no_cond:
            opt.D_input_nc = nc_img
        else:
            opt.D_input_nc = nc_img + opt.shape_nc + (nc_edge if opt.use_edge else 0) + (nc_color if opt.use_color else 0)

        ###########################################
        # set encoder initialization
        ###########################################
        # shape
        if opt.shape_encoder_type == 'default':
            opt.shape_encoder_type = opt.encoder_type
        if opt.shape_encoder_block == 'default':
            opt.shape_encoder_block = opt.encoder_block
        if opt.which_model_init_shape_encoder == 'default':
            if opt.shape_encoder_type == 'normal' and opt.shape_encoder_block == 'residual' and opt.shape_ndowns == 5 and opt.shape_nof == 128 and opt.shape_nf == 64:
                opt.which_model_init_shape_encoder = 'ED_MMDGAN_RECON_3.0'
            elif opt.shape_encoder_type == 'normal' and opt.shape_encoder_block == 'residual' and opt.shape_ndowns == 5 and opt.shape_nof == 64 and opt.shape_nf == 32:
                opt.which_model_init_shape_encoder = 'ED_MMDGAN_RECON_3.4'
            elif opt.shape_encoder_type == 'fc' and opt.shape_encoder_block == 'residual' and opt.shape_ndowns == 5 and opt.shape_nof == 128 and opt.shape_nf == 64:
                opt.which_model_init_shape_encoder = 'ED_MMDGAN_RECON_3.2'
            elif opt.shape_encoder_type == 'fc' and opt.shape_encoder_block == 'residual' and opt.shape_ndowns == 5 and opt.shape_nof == 256 and opt.shape_nf == 64:
                opt.which_model_init_shape_encoder = 'ED_MMDGAN_RECON_3.3'
            elif opt.shape_encoder_type == 'fc' and opt.shape_encoder_block == 'residual' and opt.shape_ndowns == 5 and opt.shape_nof == 128 and opt.shape_nf == 32:
                opt.which_model_init_shape_encoder = 'ED_MMDGAN_RECON_3.5'
        # edge
        if opt.edge_encoder_type == 'default':
            opt.edge_encoder_type = opt.encoder_type
        if opt.edge_encoder_block == 'default':
            opt.edge_encoder_block = opt.encoder_block
        if opt.which_model_init_edge_encoder == 'default':
            if opt.edge_shape_guided:
                if opt.edge_mode in {'outer', 'cloth'}:
                    if opt.edge_encoder_type == 'normal' and opt.edge_encoder_block == 'residual' and opt.edge_ndowns == 5 and opt.edge_nof == 128 and opt.edge_nf == 64:
                        opt.which_model_init_edge_encoder = 'ED_MMDGAN_RECON_1.0'
                    elif opt.edge_encoder_type == 'fc' and opt.edge_encoder_block == 'residual' and opt.edge_ndowns == 5 and opt.edge_nof == 128 and opt.edge_nf == 64:
                        opt.which_model_init_edge_encoder = 'ED_MMDGAN_RECON_1.2'
                    elif opt.edge_encoder_type == 'fc' and opt.edge_encoder_block == 'residual' and opt.edge_ndowns == 5 and opt.edge_nof == 256 and opt.edge_nf == 64:
                        opt.which_model_init_edge_encoder = 'ED_MMDGAN_RECON_1.3'
                else:
                    if opt.edge_encoder_type == 'normal' and opt.edge_encoder_block == 'residual' and opt.edge_ndowns == 5 and opt.edge_nof == 128 and opt.edge_nf == 64:
                        opt.which_model_init_edge_encoder = 'ED_MMDGAN_RECON_1.1'
                    elif opt.edge_encoder_type == 'normal' and opt.edge_encoder_block == 'residual' and opt.edge_ndowns == 5 and opt.edge_nof == 64 and opt.edge_nf == 32:
                        opt.which_model_init_edge_encoder = 'ED_MMDGAN_RECON_1.8'
                    elif opt.edge_encoder_type == 'fc' and opt.edge_encoder_block == 'residual' and opt.edge_ndowns == 5 and opt.edge_nof == 128 and opt.edge_nf == 64:
                        opt.which_model_init_edge_encoder = 'ED_MMDGAN_RECON_1.4'
                    elif opt.edge_encoder_type == 'fc' and opt.edge_encoder_block == 'residual' and opt.edge_ndowns == 5 and opt.edge_nof == 256 and opt.edge_nf == 64:
                        opt.which_model_init_edge_encoder = 'ED_MMDGAN_RECON_1.5'
                    elif opt.edge_encoder_type == 'fc' and opt.edge_encoder_block == 'residual' and opt.edge_ndowns == 5 and opt.edge_nof == 128 and opt.edge_nf == 32:
                        opt.which_model_init_edge_encoder = 'ED_MMDGAN_RECON_1.9'
            else:
                if opt.edge_mode in {'outer', 'cloth'}:
                    if opt.edge_encoder_type == 'normal' and opt.edge_encoder_block == 'residual' and opt.edge_ndowns == 5 and opt.edge_nof == 128 and opt.edge_nf == 64:
                        opt.which_model_init_edge_encoder = 'ED_MMDGAN_RECON_1.10'
                    elif opt.edge_encoder_type == 'normal' and opt.edge_encoder_block == 'residual' and opt.edge_ndowns == 5 and opt.edge_nof == 256 and opt.edge_nf == 64:
                        opt.which_model_init_edge_encoder = 'ED_MMDGAN_RECON_1.11'
        # color
        if opt.color_encoder_type == 'default':
            opt.color_encoder_type = opt.encoder_type
        if opt.color_encoder_block == 'default':
            opt.color_encoder_block = opt.encoder_block
        if opt.which_model_init_color_encoder == 'default':
            if opt.color_shape_guided:
                if opt.color_patch and opt.color_patch_mode == 'crop5':
                    if opt.color_encoder_type == 'normal' and opt.color_encoder_block == 'residual' and opt.color_ndowns == 5 and opt.color_nof == 128 and opt.color_nf == 64:
                        opt.which_model_init_color_encoder = 'ED_MMDGAN_RECON_2.4'
                    elif opt.color_encoder_type == 'normal' and opt.color_encoder_block == 'residual' and opt.color_ndowns == 5 and opt.color_nof == 64 and opt.color_nf == 32:
                        opt.which_model_init_color_encoder = 'ED_MMDGAN_RECON_2.11'
                    elif opt.color_encoder_type == 'fc' and opt.color_encoder_block == 'residual' and opt.color_ndowns == 5 and opt.color_nof == 128 and opt.color_nf == 64:
                        opt.which_model_init_color_encoder = 'ED_MMDGAN_RECON_2.7'
                    elif opt.color_encoder_type == 'fc' and opt.color_encoder_block == 'residual' and opt.color_ndowns == 5 and opt.color_nof == 256 and opt.color_nf == 64:
                        opt.which_model_init_color_encoder = 'ED_MMDGAN_RECON_2.8'
                    elif opt.color_encoder_type == 'fc' and opt.color_encoder_block == 'residual' and opt.color_ndowns == 5 and opt.color_nof == 128 and opt.color_nf == 32:
                        opt.which_model_init_color_encoder = 'ED_MMDGAN_RECON_2.12'
            else:
                if opt.color_patch and opt.color_patch_mode == 'crop5':
                    if opt.color_encoder_type == 'normal' and opt.color_encoder_block == 'residual' and opt.color_ndowns == 5 and opt.color_nof == 128 and opt.color_nf == 64:
                        opt.which_model_init_color_encoder = 'ED_MMDGAN_RECON_2.13'
                    elif opt.color_encoder_type == 'normal' and opt.color_encoder_block == 'residual' and opt.color_ndowns == 5 and opt.color_nof == 256 and opt.color_nf == 64:
                        opt.which_model_init_color_encoder = 'ED_MMDGAN_RECON_2.14'

        
class TrainMMGANOptions_V3(BaseMMGANOptions_V3):

    def initialize(self):

        super(TrainMMGANOptions_V3,self).initialize()
        parser = self.parser
        
        # basic
        parser.add_argument('--continue_train', action = 'store_true', default = False, help = 'coninue training from saved model')

        # train setting
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        
        # optimizer (we use Adam)
        parser.add_argument('--lr', type = float, default = 2e-4, help = 'initial learning rate')
        parser.add_argument('--lr_D', type=float, default=2e-4, help='initial learning rate for netD')
        parser.add_argument('--lr_FTN', type=float, default=2e-4, help='initial learning rate for FTN')
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
        parser.add_argument('--loss_weight_trans_feat', type=float, default=1, help = 'loss weight of feature transfer loss (feat distance)')
        parser.add_argument('--loss_weight_trans_img', type=float, default=1, help = 'loss weight of feature transfer loss (image distance)')
        parser.add_argument('--loss_weight_seg', type=float, default=1, help = 'loss weight of segmentation prediction')
        # set train
        self.is_train = True


class TestMMGANOptions_V3(BaseMMGANOptions_V3):

    def initialize(self):

        super(TestMMGANOptions_V3, self).initialize()
        parser = self.parser
        
        # test

        # set test
        self.is_train = False


        

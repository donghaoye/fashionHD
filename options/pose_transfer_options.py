from base_options import BaseOptions

class BasePoseTransferOptions(BaseOptions):
    def initialize(self):
        super(BasePoseTransferOptions, self).initialize()
        parser = self.parser
        ##############################
        # General Setting
        ##############################
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [batch|instance|none]')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
        parser.add_argument('--pavi', default = False, action = 'store_true', help = 'activate pavi log')
        parser.add_argument('--supervised', type=int, default=1, choices=[0,1], help='supervised: img_ref=img_1, pose_tar=pose_2, img_tar=img_2; unsupervised: img_ref=img_tar=img_1, pose_tar=pose_1')
        ##############################
        # Appearance Setting
        ##############################
        parser.add_argument('--appearance_type', type=str, default='image', help='appearance input format, combination ("+") of [image, limb]')
        ##############################
        # Pose Setting
        ##############################
        parser.add_argument('--pose_type', type=str, default='joint', help='pose format, combination ("+") of [joint, stickman, seg]')
        parser.add_argument('--joint_radius', type=int, default=8, help='radius of joint map')
        parser.add_argument('--joint_mode', type=str, default='binary', choices=['binary', 'gaussian'])
        parser.add_argument('--seg_bin_size', type=int, default=16, help='bin size of downsampled seg mask')
        parser.add_argument('--patch_indices', type=int, default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], nargs='+', help='indices of joint points to extract patches. see misc.pose_util.py for details')
        parser.add_argument('--patch_size', type=int, default=32, help='size of the local pathch for computing style loss')
        ##############################
        # Transformer Setting
        ##############################
        parser.add_argument('--which_model_T', type=str, default='2stage', choices=['2stage', 'unet', 'resnet', 'vunet'], help='pose transfer network architecture')
        parser.add_argument('--T_nf', type=int, default=64, help='output channel number of the first conv layer in netT')
        parser.add_argument('--output_type', type=str, default='image', help='combination of "image", "seg", ...')
        ##############################
        # Transformer Setting - VUnet
        ##############################
        parser.add_argument('--vunet_nf', type=int, default=32, help='vunet setting: channel number of the first conv layer')
        parser.add_argument('--vunet_max_nf', type=int, default=128, help='vunet setting: max channel number of mid-level conv layers')
        parser.add_argument('--vunet_n_latent_scales', type=int, default=2, help='vunet setting: layer number of latent space')
        parser.add_argument('--vunet_bottleneck_factor', type=int, default=2, help='vunet setting: the bottleneck resolution will be 2**#')
        parser.add_argument('--vunet_box_factor', type=int, default=2, help='vunet setting: the size of pose input will be reduced 2**# times')
        parser.add_argument('--vunet_activation', type=str, default='relu', choices=['relu', 'elu'], help='activation type')
        ##############################
        # Transformer Setting - 2stage
        ##############################
        parser.add_argument('--which_model_stage_1', type=str, default='PoseTransfer_4.3', help='pretrained pose transfer model as stage-1 network')
        parser.add_argument('--which_model_s2d', type=str, default='resnet', choices=['resnet', 'unet'], help='stage-2 decoder architecture')
        parser.add_argument('--s2e_nf', type=int, default=32, help='2-stage model setting: channel number of the first encoder conv layer')
        parser.add_argument('--s2e_max_nf', type=int, default=128, help='2-stage model setting: max channel number of encoder conv layers')
        parser.add_argument('--s2e_nof', type=int, default=32, help='2-stage model setting: local embedding dimension')
        parser.add_argument('--s2e_bottleneck_factor', type=int, default=2, help='2-stage model setting: the bottleneck resolution will be 2**#')
        parser.add_argument('--s2d_nf', type=int, default=64, help='2-stage model setting: channel number of the first decoder conv layer')
        parser.add_argument('--s2d_nblocks', type=int, default=6, help='2-stage model setting: resnet blocks in decoder (see networks.ResnetGenerator)')
        ##############################
        # Discriminator Setting
        ##############################
        parser.add_argument('--which_gan', type=str, default='dcgan', choices=['dcgan', 'lsgan'], help='gan loss type')
        parser.add_argument('--D_nf', type=int, default=64, help='output channel number of the first conv layer in netD')
        parser.add_argument('--D_cond', type=int, default=0, choices=[0,1], help='use conditioned discriminator')
        parser.add_argument('--pool_size', type=int, default=50, help='size of fake pool')
        ##############################
        # data setting (dataset_mode == pose_transfer_dataset)
        ##############################
        parser.add_argument('--dataset_mode', type=str, default='pose_transfer', help='type of dataset. see data/data_loader.py')
        parser.add_argument('--data_root', type=str, default='datasets/DF_Pose/')
        parser.add_argument('--fn_split', type=str, default='Label/pair_split.json')
        parser.add_argument('--img_dir', type=str, default='Img/img_df/')
        parser.add_argument('--seg_dir', type=str, default='Img/seg_df/')
        parser.add_argument('--fn_pose', type=str, default='Label/pose_label.pkl')
        parser.add_argument('--debug', action='store_true', help='debug')

    def auto_set(self):
        super(BasePoseTransferOptions, self).auto_set()
        opt = self.opt
        ###########################################
        # Add id profix
        ###########################################
        if opt.which_model_T == '2stage':
            if not opt.id.startswith('PoseTransfer_2s_'):
                opt.id = 'PoseTransfer_2s_' + opt.id
        else:
            if not opt.id.startswith('PoseTransfer_'):
                opt.id = 'PoseTransfer_' + opt.id


class TrainPoseTransferOptions(BasePoseTransferOptions):
    def initialize(self):
        super(TrainPoseTransferOptions, self).initialize()
        self.is_train = True
        parser = self.parser        
        # basic
        parser.add_argument('--continue_train', action = 'store_true', default = False, help = 'coninue training from saved model')
        # optimizer
        parser.add_argument('--lr', type = float, default = 2e-4, help = 'initial learning rate')
        parser.add_argument('--lr_D', type = float, default = 1e-4, help = 'only use lr_D for netD when loss_weight_gan > 0')
        parser.add_argument('--beta1', type = float, default = 0.5, help = 'momentum1 term for Adam')
        parser.add_argument('--beta2', type = float, default = 0.999, help = 'momentum2 term for Adam')
        # scheduler
        parser.add_argument('--lr_policy', type=str, default='step', choices = ['step', 'plateau', 'lambda'], help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--niter', type = int, default=30, help = '# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_decay', type=int, default=10, help='multiply by a gamma every lr_decay_interval epochs')
        parser.add_argument('--lr_gamma', type = float, default = 0.1, help='lr decay rate')
        parser.add_argument('--display_freq', type = int, default = 10, help='frequency of showing training results on screen')
        parser.add_argument('--test_epoch_freq', type = int, default = 1, help='frequency of testing model')
        parser.add_argument('--save_epoch_freq', type = int, default = 5, help='frequency of saving model to disk' )
        parser.add_argument('--vis_epoch_freq', type = int, default = 1, help='frequency of visualizing generated images')
        parser.add_argument('--check_grad_freq', type = int, default = 100, help = 'frequency of checking gradient of each loss')
        parser.add_argument('--max_n_vis', type = int, default = 64, help='max number of visualized images')
        # loss weights
        parser.add_argument('--loss_weight_L1', type=float, default=1.)
        parser.add_argument('--loss_weight_content', type=float, default=1.)
        parser.add_argument('--loss_weight_style', type=float, default=0., help='set loss_weight_style > 0 to enable global style loss')
        parser.add_argument('--loss_weight_patch_style', type=float, default=0., help='set loss_weight_patch_style > 0 to enable patch style loss')
        parser.add_argument('--loss_weight_gan', type=float, default=0., help='set loss_weight_gan > 0 to enable GAN loss')
        parser.add_argument('--loss_weight_kl', type=float, default=1e-6, help='vunet setting: kl loss weight')
        parser.add_argument('--loss_weight_seg', type=float, default=0.1, help='weight of cross entropy loss on additional segmentation outputs')
        # train 2-stage model
        parser.add_argument('--train_s1', type=int, default=0, choices=[0,1], help='set 1 to jointly train stage-1 and stage-2 networks')
        parser.add_argument('--lr_s1', type=float, default=2e-5, help='initial learning rate for stage-1 net when joint training')

class TestPoseTransferOptions(BasePoseTransferOptions):
    def initialize(self):
        super(TestPoseTransferOptions, self).initialize()
        self.is_train = False
        parser = self.parser

        parser.add_argument('--nbatch', type=int, default=-1, help='set number of minibatch used for test')
        parser.add_argument('--save_output', action='store_true', help='save output images in the folder exp_dir/output/')
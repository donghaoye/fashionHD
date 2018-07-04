from base_options import BaseOptions

class BasePoseParsingOptions(BaseOptions):
    def initialize(self):
        super(BasePoseParsingOptions, self).initialize()
        parser = self.parser
        ##############################
        # General Setting
        ##############################
        parser.add_argument('--batch_size', type = int, default = 16, help = 'batch size')
        parser.add_argument('--pavi', default = False, action = 'store_true', help = 'activate pavi log')
        ##############################
        # Model Setting
        ##############################
        parser.add_argument('--which_model_PP', type=str, default='unet', choices=['unet', 'resnet'], help='model architecture')
        parser.add_argument('--pp_nf', type=int, default=32, help='')
        parser.add_argument('--pp_nblocks', type=int, default=6, help='')
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [batch|instance|none]')
        ##############################
        # Pose Setting
        ##############################
        parser.add_argument('--pp_pose_type', type=str, default='seg+joint')
        parser.add_argument('--seg_nc', type=int, default=8, help='number of segmentation classes, 7 for ATR and 8 for LIP')
        parser.add_argument('--joint_nc', type=int, default=18, help='number of joint keys, 18 for OpenPose setting')
        ##############################
        # Data Setting
        ##############################
        parser.add_argument('--dataset_mode', type=str, default='pose_parsing', help='type of dataset. see data/data_loader.py')
        parser.add_argument('--data_root', type=str, default='datasets/DF_Pose/')
        parser.add_argument('--fn_split', type=str, default='Label/split.json')
        parser.add_argument('--img_dir', type=str, default='Img/img_df/')
        parser.add_argument('--seg_dir', type=str, default='Img/seg-lip_df_revised/')
        parser.add_argument('--fn_pose', type=str, default='Label/pose_label.pkl')
        parser.add_argument('--debug', action='store_true', help='debug')
        
    def auto_set(self):
        super(BasePoseParsingOptions, self).auto_set()
        opt = self.opt
        ###########################################
        # Add id profix
        ###########################################
        if not opt.id.startswith('PoseParsing_'):
            opt.id = 'PoseParsing_' + opt.id

class TrainPoseParsingOptions(BasePoseParsingOptions):
    def initialize(self):
        super(TrainPoseParsingOptions, self).initialize()
        self.is_train = True
        parser = self.parser
        # basic
        parser.add_argument('--continue_train', action = 'store_true', default = False, help = 'coninue training from saved model')
        # optimizer
        parser.add_argument('--lr', type = float, default = 1e-5, help = 'initial learning rate')
        parser.add_argument('--beta1', type = float, default = 0.9, help = 'momentum1 term for Adam')
        parser.add_argument('--beta2', type = float, default = 0.999, help = 'momentum2 term for Adam')
        # scheduler
        parser.add_argument('--lr_policy', type=str, default='step', choices = ['step', 'plateau', 'lambda'], help='learning rate policy: lambda|step|plateau')
        # parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--niter', type = int, default=30, help = '# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_decay', type=int, default=10, help='multiply by a gamma every lr_decay_interval epochs')
        parser.add_argument('--lr_gamma', type = float, default = 0.1, help='lr decay rate')
        parser.add_argument('--display_freq', type = int, default = 10, help='frequency of showing training results on screen')
        parser.add_argument('--test_epoch_freq', type = int, default = 1, help='frequency of testing model')
        parser.add_argument('--save_epoch_freq', type = int, default = 5, help='frequency of saving model to disk' )
        parser.add_argument('--vis_epoch_freq', type = int, default = 1, help='frequency of visualizing generated images')
        parser.add_argument('--check_grad_freq', type = int, default = 100, help = 'frequency of checking gradient of each loss')
        parser.add_argument('--nvis', type = int, default = 64, help='number of visualized images')
        # loss setting
        parser.add_argument('--loss_weight_seg', type=float, default=1., help='segmentation loss weight (cross entropy loss)')
        parser.add_argument('--loss_weight_joint', type=float, default=1., help='pose loss weight (mse loss)')

class TestPoseParsingOptions(BasePoseParsingOptions):
    def initialize(self):
        super(TestPoseParsingOptions, self).initialize()
        self.is_train = False
        parser = self.parser
        parser.add_argument('--nbatch', type=int, default=-1, help='set number of minibatch used for test')
        parser.add_argument('--save_output', action='store_true', help='save output images in the folder exp_dir/test/')
        parser.add_argument('--vis_only', action='store_true', help='only viusal')
        parser.add_argument('--nvis', type = int, default = 64, help='number of visualized images')



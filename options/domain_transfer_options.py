from base_options import BaseOptions

class BaseDomainTransferOptions(BaseOptions):
    def initialize(self):
        super(BaseDomainTransferOptions, self).initialize()
        parser = self.parser
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [batch|instance|none]')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
        parser.add_argument('--pavi', default = False, action = 'store_true', help = 'activate pavi log')
        ##############################
        # Pose Setting
        ##############################
        parser.add_argument('--pose_type', type=str, default='joint', help='pose format, combination ("+") of [joint, stickman, seg]')
        parser.add_argument('--joint_radius', type=int, default=8, help='radius of joint map')
        parser.add_argument('--joint_mode', type=str, default='binary', choices=['binary', 'gaussian'])
        parser.add_argument('--seg_bin_size', type=int, default=16, help='bin size of downsampled seg mask')        
        parser.add_argument('--patch_size', type=int, default=32, help='size of the local pathch for computing style loss')
        ##############################
        # data setting
        ##############################
        parser.add_argument('--dataset_mode', type=str, default='domain_transfer', help='type of dataset. see data/data_loader.py')
        parser.add_argument('--data_root', type=str, default='datasets/Zalando/')
        parser.add_argument('--fn_split', type=str, default='Split/zalando_split.json')
        parser.add_argument('--img_dir_1', type=str, default='Img/img_zalando_person/')
        parser.add_argument('--img_dir_2', type=str, default='Img/img_zalando_cloth/')
        parser.add_argument('--seg_dir', type=str, default='Img/seg_zalando_256/')
        parser.add_argument('--fn_pose', type=str, default='Label/zalando_pose_label_256.pkl')
        parser.add_argument('--debug', action='store_true', help='debug')

    def auto_set(self):
        super(BaseDomainTransferOptions, self).auto_set()
        opt = self.opt
        ###########################################
        # Add id profix
        ###########################################
        if not opt.id.startswith('DomainTransfer_'):
            opt.id = 'DomainTransfer_' + opt.id


class TrainDomainTransferOptions(BaseDomainTransferOptions):
    def initialize(self):
        super(TrainDomainTransferOptions, self).initialize()
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
        parser.add_argument('--max_n_vis', type = int, default = 32, help='max number of visualized images')
        # loss weights
        parser.add_argument('--loss_weight_L1', type=float, default=1.)
        parser.add_argument('--loss_weight_content', type=float, default=1.)
        parser.add_argument('--loss_weight_style', type=float, default=0., help='set loss_weight_style > 0 to enable global style loss')
        # parser.add_argument('--loss_weight_patch_style', type=float, default=0., help='set loss_weight_patch_style > 0 to enable patch style loss')
        parser.add_argument('--loss_weight_gan', type=float, default=0., help='set loss_weight_gan > 0 to enable GAN loss')
        parser.add_argument('--loss_weight_kl', type=float, default=1e-6, help='vunet setting: kl loss weight')

class TestDomainTransferOptions(BaseDomainTransferOptions):
    def initialize(self):
        super(TestDomainTransferOptions, self).initialize()
        self.is_train = False
        parser = self.parser

        parser.add_argument('--nbatch', type=int, default=-1, help='set number of minibatch used for test')
        parser.add_argument('--save_output', action='store_true', help='save output images in the folder exp_dir/output/')

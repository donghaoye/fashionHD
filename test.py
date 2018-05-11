from __future__ import division

from models.two_stage_pose_transfer_model import TwoStagePoseTransferModel
from options.pose_transfer_options import TrainPoseTransferOptions

opt = TrainPoseTransferOptions().parse()
opt.which_model_s2d = 'unet'
model = TwoStagePoseTransferModel()
model.initialize(opt)


from __future__ import division, print_function

from models.attribute_encoder import AttributeEncoder
from data.data_loader import CreateDataLoader
from options.attribute_options import TrainAttributeOptions
from options.base_options import opt_to_str
from models.networks import MeanAP

import os
import sys
import time
from util.pavi import PaviClient
import util.io as io
# Todo: implement visualizer
# from util.visualizer import AttributeVisualizer


opt = TrainAttributeOptions().parse()
train_loader = CreateDataLoader(opt)
train_dataset_size = len(train_loader)

val_loader = CreateDataLoader(opt, is_train = False)
val_dataset_size = len(val_loader)

model = AttributeEncoder()
model.initialize(opt)


# Pavi client
if opt.pavi:
    pavi = PaviClient(username = 'ly015', password = '123456')
    pavi.connect(model_name = opt.id, info = {'session_text': opt_to_str(opt)})

total_steps = 0
f_log = open(os.path.join('checkpoints', opt.id, 'train_log.txt'), 'w')

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()

    epoch_samples = 0

    for i, data in enumerate(train_loader):
        iter_start_time = time.time()
        total_steps += 1
        
        epoch_samples += opt.batch_size
        

        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            train_error = model.get_current_errors()
            t = time.time() - iter_start_time
            log = '[%s] Train Epoch %d, [%d/%d (%.2f%%)] t_cost: %.1f   lr: %.3e   ' % \
                    (opt.id, epoch, epoch_samples, len(train_loader.dataset), 100.*epoch_samples/len(train_loader.dataset), t, model.optimizers[0].param_groups[0]['lr'])

            log += '   '.join(['%s: %.6f' % (name, value) for name, value in train_error.iteritems()])

            print(log)
            print(log, file = f_log)

            if opt.pavi:
                pavi_log = {
                    'loss_attr': train_error['loss_attr'],
                }
                pavi.log(phase = 'train', iter_num = total_steps, outputs = pavi_log)


    if epoch % opt.test_epoch_freq == 0:
        test_start_time = time.time()
        crit_ap = MeanAP()
        _ = model.get_current_errors()# clean loss buffer

        for i, data in enumerate(val_loader):
            model.set_input(data)
            model.test()
            crit_ap.add(model.output_prob, model.input_label)

            print('\rTesting %d/%d (%.2f%%)' % (i, len(val_loader), 100.*i/len(val_loader)), end = '')
            sys.stdout.flush()

        mean_ap, ap_list = crit_ap.compute_mean_ap()
        balance_ap, balance_ap_list = crit_ap.compute_balance_ap()
        
        t = time.time() - test_start_time
        test_error = model.get_current_errors()
        log = '[%s] Test Epoch %d, time taken: %d sec,  loss: %.6f   meanAP: %.4f' % \
            (opt.id, epoch, t, test_error['loss_attr'], mean_ap)

        crit_ap.clear()
        print(log)
        print(log, file = f_log)

        if opt.pavi:
            pavi_log = {
                'loss_attr': test_error['loss_attr'],
                'mean_ap_upper': float(mean_ap),
                'balance_ap_upper': float(balance_ap),
            }
            pavi.log(phase = 'test', iter_num = total_steps, outputs = pavi_log)


        # save model
        if epoch % opt.save_epoch_freq == 0:
            log = 'saving the model at the end of epoch %d, iters %d' % (epoch, total_steps)
            print(log)
            print(log, file = f_log)
            model.save(epoch)
            model.save('latest')

            test_info = {
                'mean_ap': mean_ap,
                'ap_list': ap_list,
                'balance_ap': balance_ap,
                'balance_ap_list': balance_ap_list,
            }
            io.save_data(test_info, os.path.join('checkpoints', opt.id, 'test_info_%d.pkl'%epoch))


    log = 'end of epoch %d / %d, time taken: %d sec' % \
        (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
    print(log)
    print(log, file = f_log)
    model.update_learning_rate()


f_log.close()







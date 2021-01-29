import time
from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CreateDataLoader
from model.reflection_removal import ReflectionRemovalModel
from util import util
import numpy as np
import os
import scipy.io as io
import torch


opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = ReflectionRemovalModel()
model.initialize(opt)
total_steps = 0

loss_fake_transmission = []
loss_fake_reflection = []
loss_W = []
loss_IO = []
loss_fake_transmission_grad = []

myloss = {}
myloss['loss_G'] = []
myloss['loss_D'] = []
myloss['loss_t'] = []
myloss['loss_r'] = []
myloss['loss_w'] = []
myloss['loss_rec'] = []
myloss['loss_tg'] = []
myloss['loss_wg'] = []
myloss['loss_trg'] = []
myloss['loss_n'] = []


def print_current_errors(epoch, i,totoal_num, errors, t, t_data, log_name):
    message = '(epoch: %d, iters: %d/%d) ' % (epoch, i,totoal_num)
    for k, v in errors.items():
        myloss[k].append(v)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, sum(myloss[k])/len(myloss[k]))
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

    myloss['loss_G'] = []
    myloss['loss_D'] = []
    myloss['loss_t'] = []
    myloss['loss_r'] = []
    myloss['loss_w'] = []
    myloss['loss_rec'] = []
    myloss['loss_tg'] = []
    myloss['loss_wg'] = []
    myloss['loss_trg'] = []
    myloss['loss_n'] = []

    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0
    totoal_num = len(dataset)

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        if total_steps % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % 1000 == 0:
            img_dir = os.path.join(opt.checkpoints_dir, opt.which_type, 'images')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            for label, image_numpy in model.get_current_visuals_train().items():
                if label=='fake_norm' or label=='real_norm':
                    result1 = np.array(image_numpy.cpu().data)
                    np.savetxt(os.path.join(img_dir, 'epoch%.3d_%s.txt' % (epoch, label)),result1)
                else:
                    img_path = os.path.join(img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                    util.save_image(image_numpy, img_path)
        if total_steps % 100 == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batch_size

            log_name = os.path.join(opt.checkpoints_dir, opt.which_type, 'loss_log.txt')
            print_current_errors(epoch, epoch_iter,totoal_num, errors, t, t_data, log_name)
            
        iter_data_time = time.time()

    loss_fake_transmission.append(errors['loss_t'])
    loss_fake_reflection.append(errors['loss_r'])
    loss_W.append(errors['loss_w'])
    loss_IO.append(errors['loss_rec'])
    loss_fake_transmission_grad.append(errors['loss_tg'])

    np.save(os.path.join(opt.checkpoints_dir, opt.which_type, 'loss_fake_transmission'), np.asarray(loss_fake_transmission))
    np.save(os.path.join(opt.checkpoints_dir, opt.which_type, 'loss_fake_reflection'), np.asarray(loss_fake_reflection))
    np.save(os.path.join(opt.checkpoints_dir, opt.which_type, 'loss_W'), np.asarray(loss_W))
    np.save(os.path.join(opt.checkpoints_dir, opt.which_type, 'loss_IO'), np.asarray(loss_IO))
    np.save(os.path.join(opt.checkpoints_dir, opt.which_type, 'loss_fake_transmission_grad'), np.asarray(loss_fake_transmission_grad))

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

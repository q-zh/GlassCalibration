import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import torch.nn as nn
 

class ReflectionRemovalModel(BaseModel):
    def name(self):
        return 'ReflectionRemovalModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # load/define networks
        self.gpu_ids = opt.gpu_ids
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netD = networks.define_D(opt.input_nc, self.gpu_ids)
        self.get_gradient = networks.ImageGradient(self.gpu_ids)
        self.weight = torch.tensor([0.6, 0.2, 0.2]).cuda(self.gpu_ids[0])*3
        self.fake_rate = 0.2
        self.opt = opt
        
        if len(self.gpu_ids) > 0:
            self.get_gradient.cuda(self.gpu_ids[0])

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lrd, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_C = input['C']
        if self.opt.phase == 'train':
            input_A = input['A']
            input_B = input['B']
            input_W = input['W']
            input_norm = input['norm']
        if len(self.gpu_ids) > 0:
            input_C = input_C.cuda(self.gpu_ids[0], non_blocking=True)
            if self.opt.phase == 'train':
                input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
                input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
                input_W = input_W.cuda(self.gpu_ids[0], non_blocking=True)
                input_norm = input_norm.cuda(self.gpu_ids[0], non_blocking=True)
        self.input_C = input_C
        if self.opt.phase == 'train':
            self.input_A = input_A
            self.input_B = input_B
            self.input_W = input_W
            self.input_norm = input_norm
        self.image_filename = input['filename']

    def forward(self):
        self.real_C = self.input_C
        if self.opt.phase == 'train':
            self.real_transmission = self.input_A
            self.real_reflection = self.input_B
            self.real_W = self.input_W
            self.real_norm = self.input_norm

    def test(self):
        real_C = self.input_C
        fake_transmission, fake_reflection, fake_W = self.netG(real_C)
        fake_norm1,_,_,_ = self.netD(fake_W)

        synthetic_C = fake_transmission * (1-fake_W) + fake_reflection * fake_W

        self.fake_transmission = fake_transmission.data
        self.fake_reflection = fake_reflection.data
        self.fake_norm = fake_norm1.data
        self.fake_W = fake_W.data
        self.synthetic_C = synthetic_C.data
        self.real_C = real_C.data

    # get image paths
    def get_image_paths(self):
        return self.opt.dataroot 


    def compute_exclusion_loss(self, img1,img2,level=1):
        gradx_loss = []
        grady_loss = []

        for l in range(level):
            gradx1, grady1 = self.get_gradient(img1)
            gradx2, grady2 = self.get_gradient(img2)
            alphax = 2.0 * (torch.mean(abs(gradx1))+1e-12 )/ (torch.mean(abs(gradx2))+1e-12)
            alphay = 2.0 * (torch.mean(abs(grady1))+1e-12) / (torch.mean(abs(grady2))+1e-12)
            gradx1h =  torch.tanh(gradx1)
            gradx2h =  torch.tanh(gradx2 * alphax)#.cuda(self.gpu_ids[0])
            grady1h =  torch.tanh(grady1)
            grady2h =  torch.tanh(grady2 * alphay)
            gradx_loss.append( ( torch.mean(torch.mul(gradx1h**2.0, gradx2h**2.0) )+ 1e-12)**0.25 )
            grady_loss.append( (torch.mean(torch.mul(grady1h**2.0 , grady2h** 2.0) )+ 1e-12)**0.25)

            # grady_loss.append(torch.norm(torch.mul(grady1h, grady2h)))
            m = torch.nn.AdaptiveAvgPool2d( (int(img1.shape[2] / 2), int(img1.shape[3] / 2)) ).cuda(self.gpu_ids[0])
            img1 = m(img1)
            img2 = m(img2)

        grad_loss=(sum(gradx_loss) / 3.0) + (sum(grady_loss) / 3.0) / 2.0
        # print('graloss',grad_loss)
        return grad_loss

    def backward_G(self):

        fake_transmission, fake_reflection, fake_W = self.netG(self.real_C)
        fake_transmission_grad_x, fake_transmission_grad_y = self.get_gradient(fake_transmission)
        real_transmission_grad_x, real_transmission_grad_y = self.get_gradient(self.real_transmission)

        synthetic_C = fake_transmission * (1-fake_W) + fake_reflection * fake_W
        loss_fake_transmission = self.criterionL1(fake_transmission, self.real_transmission) 
        loss_fake_reflection = self.criterionL1(fake_reflection, self.real_reflection) 
        loss_fake_transmission_grad = self.criterionL1(fake_transmission_grad_x, real_transmission_grad_x)  \
                                      + self.criterionL1(fake_transmission_grad_y, real_transmission_grad_y)

        # smooth_y_W = self.criterionL2(fake_W[:, :, 1:, :], fake_W.detach()[:, :, :-1, :])
        # smooth_x_W = self.criterionL2(fake_W[:, :, :, 1:], fake_W.detach()[:, :, :, :-1])
        loss_Smooth_W = 0 #smooth_y_W + smooth_x_W

        loss_W = self.criterionL1(fake_W, self.real_W) 
        loss_IO = self.criterionL1(synthetic_C, self.real_C) 

        fake_norm1, fake_norm2, fake_norm3, fake_norm4 = self.netD(fake_W)
        

        loss_D = self.computeLossD(fake_norm1, fake_norm2, fake_norm3, fake_norm4, self.real_norm)

        loss_Sparse = 0 #self.compute_exclusion_loss(fake_transmission,fake_reflection, level=3)

        loss_G = loss_fake_transmission*100.0 + loss_fake_reflection*50.0 + loss_W* 100.0 + loss_IO*50.0 \
               +  loss_fake_transmission_grad*50.0 + loss_D #+ loss_Sparse +loss_Smooth_W*100.0 

        
        loss_G.backward()

        self.fake_transmission = fake_transmission.data
        self.fake_reflection = fake_reflection.data
        self.fake_W = fake_W.data
        self.synthetic_C = synthetic_C.data
        self.fake_WforD = fake_W
        self.fake_norm = fake_norm1.data

        self.loss_t=loss_fake_transmission.item()
        self.loss_r=loss_fake_reflection.item()
        self.loss_w=loss_W.item()
        self.loss_rec=loss_IO.item()
        self.loss_tg=loss_fake_transmission_grad.item()
        self.loss_n=loss_D.item()
        self.loss_wg=0#loss_Smooth_W.item()
        self.loss_G = loss_G.item()
        self.loss_spa = 0 #loss_Sparse.item()

    def computeLossD(self, fake_norm1, fake_norm2, fake_norm3, fake_norm4, real_norm):
        loss_D1 = torch.mean(self.weight*torch.abs(fake_norm1-real_norm))
        loss_D2 = torch.mean(self.weight*torch.abs(fake_norm2-real_norm))
        loss_D3 = torch.mean(self.weight*torch.abs(fake_norm3-real_norm))
        loss_D4 = torch.mean(self.weight*torch.abs(fake_norm4-real_norm))
        return (loss_D1+loss_D2+loss_D3+loss_D4)*0.25

    def backward_D(self):
        fake_norm1, fake_norm2, fake_norm3, fake_norm4 = self.netD(self.fake_WforD.detach())
        real_norm1, real_norm2, real_norm3, real_norm4 = self.netD(self.real_W)
        loss_D1 = self.computeLossD(fake_norm1, fake_norm2, fake_norm3, fake_norm4, self.real_norm)
        loss_D2 = self.computeLossD(real_norm1, real_norm2, real_norm3, real_norm4, self.real_norm)
        loss_D= (loss_D1 +loss_D2)*0.5
        loss_D.backward()
        
        self.loss_D = loss_D.item()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('loss_G', self.loss_G),
                                  ('loss_D', self.loss_D),
                                  ('loss_t', self.loss_t),
                                  ('loss_r', self.loss_r),
                                  ('loss_w', self.loss_w),
                                  ('loss_rec', self.loss_rec),
                                  ('loss_tg', self.loss_tg),
                                  ('loss_wg', self.loss_wg),
                                  ('loss_trg', self.loss_spa),
                                  ('loss_n', self.loss_n)])
        return ret_errors

    def get_current_visuals_train(self):
        real_transmission = util.tensor2im(self.real_transmission)
        real_reflection = util.tensor2im(self.real_reflection)
        real_C = util.tensor2im(self.real_C)
       

        fake_transmission = util.tensor2im(self.fake_transmission)
        fake_reflection = util.tensor2im(self.fake_reflection)
        fake_norm = self.fake_norm
        fake_W = util.tensor2imW(self.fake_W.repeat([1,3,1,1]))
        real_norm = self.real_norm
        real_W = util.tensor2imW(self.real_W.repeat([1,3,1,1]))


        synthetic_C = util.tensor2im(self.synthetic_C)

        ret_visuals = OrderedDict([('fake_transmission', fake_transmission), ('fake_reflection', fake_reflection), ('real_C', real_C), 
            ('fake_norm', fake_norm),  ('real_norm', real_norm), ('real_transmission', real_transmission), 
            ('real_reflection', real_reflection), ('synthetic_C', synthetic_C), ('fake_W', fake_W), ('real_W', real_W)])
                                # 
        return ret_visuals
    def get_current_norm(self):
        return self.fake_norm
    def get_current_visuals_test(self):
        fake_transmission = util.tensor2im(self.fake_transmission)
#        fake_reflection = util.tensor2im(self.fake_reflection)
        real_C = util.tensor2im(self.real_C)
        synthetic_C = util.tensor2im(self.synthetic_C)

        ret_visuals = OrderedDict([('fake_transmission', fake_transmission), ('real_C', real_C), ('synthetic_C', synthetic_C)])
#                                    ('fake_reflection', fake_reflection)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

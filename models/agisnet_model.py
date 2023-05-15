import torch
from .base_model import BaseModel
from . import networks
from .vgg import VGG19
import random


class AGISNetModel(BaseModel):
    def name(self):
        return 'AGISNetModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        if opt.isTrain:
            assert opt.batch_size % 2 == 0  # load two images at one time.

        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_L1', 'G_L1_B', 'G_CX', 'G_CX_B', 'G_GAN', 'G_GAN_B', 'D', 'D_B',
                           'G_L1_val', 'G_L1_B_val', 'local_adv']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # It is up to the direction AtoB or BtoC or AtoC
        self.dirsection = opt.direction

        # AGISNet model only support AtoC now, BtoC and AtoB need to do
        # BicycleGAN model supports all
        assert(self.dirsection == 'AtoC')
        self.visual_names = ['real_A', 'real_B', 'fake_B', 'real_C', 'real_C_l', 'fake_C']
        # specify the models you want to save to the disk.
        # The program will call base_model.save_networks and base_model.load_networks
        # D for color
        use_D = opt.isTrain and opt.lambda_GAN > 0.0


        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, self.opt.nencode, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type,
                                      gpu_ids=self.gpu_ids, where_add=self.opt.where_add, upsample=opt.upsample, wo_skip=self.opt.wo_skip, cross=self.opt.cross)

        D_output_nc = (opt.input_nc + opt.output_nc) if opt.conditional_D else opt.output_nc
        use_sigmoid = opt.gan_mode == 'dcgan'
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          use_sigmoid=use_sigmoid, init_type=opt.init_type,
                                          num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)



        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(mse_loss=not use_sigmoid).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction='none')

            # Contextual Loss
            self.criterionCX = networks.CXLoss(sigma=0.5).to(self.device)
            self.vgg19 = VGG19().to(self.device)
            self.vgg19.load_model(self.opt.vgg)
            self.vgg19.eval()
            self.vgg_layers = ['conv3_3', 'conv4_2']

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)


    def is_train(self):
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)  # A is the base font
        self.real_B = input['B'].to(self.device)  # B is the gray shape
        self.real_C = input['C'].to(self.device)  # C is the color font
        self.real_Shapes = input['Shapes'].to(self.device)
        self.real_Colors = input['Colors'].to(self.device)  # Colors is reference color characters
        self.real_Colors_CA = input['Colors_CA'].to(self.device)
        self.label = input['label'].to(self.device)


    def test(self):
        with torch.no_grad():

            self.fake_C, self.x3, self.x4 = self.netG(self.real_A, self.real_Shapes, self.real_Colors, self.real_Colors_CA)
            test_l1_loss = torch.nn.L1Loss()(self.fake_C, self.real_C)
            test_ltec_loss = torch.nn.L1Loss()(self.x3, self.x4)

            self.vgg_fake_C = self.vgg19(self.fake_C)
            self.vgg_real_C = self.vgg19(self.real_C)
            self.loss_G_CX = 0.0
            for l in self.vgg_layers:
                # symmetric contextual loss
                cx, cx_batch = self.criterionCX(self.vgg_real_C[l], self.vgg_fake_C[l])
                self.loss_G_CX += torch.sum(cx_batch * self.label)


            return self.real_A, self.fake_C, self.real_C, test_l1_loss, test_ltec_loss, self.loss_G_CX


    def validate(self):
        with torch.no_grad():
            self.fake_C, self.fake_B = self.netG(self.real_A, self.real_Colors)
            self.loss_G_L1_val = torch.nn.functional.l1_loss(self.fake_C, self.real_C)
            self.loss_G_L1_B_val = torch.nn.functional.l1_loss(self.fake_B, self.real_B)
            return self.real_A, self.fake_B, self.real_B, self.fake_C, self.real_C, \
                self.loss_G_L1_B_val, self.loss_G_L1_val

    def train(self):
        for name in self.model_names:
            model_name = 'net' + name
            getattr(self, model_name).train()

    def forward(self):
        # generate fake_C
        self.fake_C, self.x3, self.x4 = self.netG(self.real_A, self.real_Shapes, self.real_Colors, self.real_Colors_CA)

        self.vgg_fake_C = self.vgg19(self.fake_C)
        self.vgg_real_C = self.vgg19(self.real_C)

        if self.opt.conditional_D:  # tedious conditoinal data
            self.fake_data_C = torch.cat([self.real_A, self.fake_C], 1)
            self.real_data_C = torch.cat([self.real_A, self.real_C], 1)



        if self.opt.conditional_D:   # tedious conditoinal data
            self.fake_data_C = torch.cat([self.real_A, self.fake_C], 1)


    def backward_D(self, netD, real, fake, blur=None):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake.detach())
        # real
        pred_real = netD(real)
        # blur
        loss_D_blur = 0.0
        if blur is not None:
            pred_blur = netD(blur)
            loss_D_blur, _ = self.criterionGAN(pred_blur, False)

        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real + loss_D_blur
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real, loss_D_blur]

    def backward_G_GAN(self, fake, net=None, ll=0.0):
        if ll > 0.0:
            pred_fake = net(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_G(self):
        # 1. G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_C, self.netD, self.opt.lambda_GAN)

        # 2. reconstruction |fake_C-real_C| |fake_B-real_B|
        self.loss_G_L1 = 0.0
        if self.opt.lambda_L1 > 0.0:
            L1 = torch.mean(torch.mean(torch.mean(
                self.criterionL1(self.fake_C, self.real_C), dim=1), dim=1), dim=1)
            self.loss_G_L1 = torch.sum(self.label * L1) * self.opt.lambda_L1

        self.loss_G_Ltec = 0.0
        if self.opt.lambda_Ltec > 0.0:
            Ltec = torch.mean(torch.mean(torch.mean(
                self.criterionL1(self.x3, self.x4), dim=1), dim=1), dim=1)
            self.loss_G_Ltec = torch.sum(self.label * Ltec) * self.opt.lambda_Ltec


        self.loss_G_CX = 0.0
        self.loss_G_CX_B = 0.0
        if self.opt.lambda_CX > 0.0:
            for l in self.vgg_layers:
                # symmetric contextual loss
                cx, cx_batch = self.criterionCX(self.vgg_real_C[l], self.vgg_fake_C[l])
                self.loss_G_CX += torch.sum(cx_batch * self.label) * self.opt.lambda_CX

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_CX + self.loss_G_Ltec

        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        # update D
        if self.opt.lambda_GAN > 0.0:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_C, self.fake_data_C)
            self.optimizer_D.step()

    def update_G(self):
        # update dual net G
        self.set_requires_grad(self.netD, False)

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G()
        self.update_D()



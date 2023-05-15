import os
import time

from data import CreateDataLoader
from models import create_model
from options.train_options import TrainOptions
from util import html
import torchvision.utils as vutils
if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.name = 'new_version'

    par = [


        {'save_path': 'SP_SSKS-27', 'A_paths': 'heiti', 'B_paths': 'kaiti', 'CA_paths': 'kaiti-04',
         'C_paths': 'kaiti-04', 'lr': 0.0002, 'nencode': 1,
         'lambda_L1': 100, 'lambda_CX': 25, 'lambda_GAN': 1, 'lambda_Ltec': 0,
         'niter': 250, 'niter_decay': 50, 'cross': [1, 1, 1, 2, 2, 2], 'wo_skip': False},

        {'save_path': '2-2kaiti04', 'A_paths': 'heiti', 'B_paths': 'kaiti', 'CA_paths': 'kaiti-04',
         'C_paths': 'kaiti-04', 'lr': 0.0002, 'nencode': 2,
         'lambda_L1': 100, 'lambda_CX': 25, 'lambda_GAN': 1, 'lambda_Ltec': 0,
         'niter': 250, 'niter_decay': 50, 'cross': [1, 1, 1, 2, 2, 2], 'wo_skip': False},

        {'save_path': '4-4kaiti04', 'A_paths': 'heiti', 'B_paths': 'kaiti', 'CA_paths': 'kaiti-04',
         'C_paths': 'kaiti-04', 'lr': 0.0002, 'nencode': 4,
         'lambda_L1': 100, 'lambda_CX': 25, 'lambda_GAN': 1, 'lambda_Ltec': 0,
         'niter': 150, 'niter_decay': 50, 'cross': [1, 1, 1, 2, 2, 2], 'wo_skip': False},

        {'save_path': '8-8kaiti04', 'A_paths': 'heiti', 'B_paths': 'kaiti', 'CA_paths': 'kaiti-04',
         'C_paths': 'kaiti-04', 'lr': 0.0002, 'nencode': 8,
         'lambda_L1': 100, 'lambda_CX': 25, 'lambda_GAN': 1, 'lambda_Ltec': 0,
         'niter': 150, 'niter_decay': 50, 'cross': [1, 1, 1, 2, 2, 2], 'wo_skip': False},



    ]
    for index in range(len(par)):
        opt.save_path = par[index]['save_path']
        opt.A_paths = par[index]['A_paths']
        opt.B_paths = par[index]['B_paths']
        opt.CA_paths = par[index]['CA_paths']
        opt.C_paths = par[index]['C_paths']
        opt.nencode = par[index]['nencode']
        opt.lr = par[index]['lr']
        opt.lambda_L1 = par[index]['lambda_L1']
        opt.lambda_CX = par[index]['lambda_CX']
        opt.lambda_GAN = par[index]['lambda_GAN']
        opt.lambda_Ltec = par[index]['lambda_Ltec']
        opt.niter = par[index]['niter']
        opt.niter_decay = par[index]['niter_decay']
        opt.cross = par[index]['cross']
        opt.wo_skip = par[index]['wo_skip']
        opt.dataroot = '/mnt/lx/ssaf/train'
        opt.vgg = '../vgg19-dcbb9e9d.pth'
        opt.nef = 64
        opt.ndf = 64
        opt.ngf = 64
        opt.batch_size = 8
        opt.threadnum = 8
        data_loader = CreateDataLoader(opt)
        data_loader_test = CreateDataLoader(opt, "test")
        dataset = data_loader.load_data()
        dataset_test = data_loader_test.load_data()
        dataset_size = len(data_loader)
        dataset_size_test = len(data_loader_test)
        print('#training images = %d' % dataset_size)
        print('#training images = %d' % dataset_size_test)

        model = create_model(opt)
        model.setup(opt)
        total_steps = 0

        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0
            model.train()
            for i, data in enumerate(dataset):
                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size

                model.set_input(data)

                model.optimize_parameters()


                iter_data_time = time.time()
            ###############################################################################################
            if epoch % 50 == 0:

                model.eval()
                web_dir = os.path.join(opt.results_dir, opt.save_path, str(epoch))
                webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class = %s' % (opt.name, 'test', opt.name))

                l1_loss_file = os.path.join(opt.results_dir, opt.save_path, "l1_loss.txt")
                cnt = 0
                mean_l1_loss = 0.0
                mean_ltec_loss = 0.0
                mean_cx_loss = 0.0
                # test stage
                for i, data in enumerate(dataset_test):
                    model.set_input(data)
                    A = data['ABC_path'][0]
                    file_name = A.split('/')[-1].split('.')[0]
                    #print('process input image %3.3d/%3.3d' % (i, opt.num_test))
                    real_A, fake_out, real_out, l1_loss, ltec_loss, cx_loss = model.test()
                    mean_l1_loss += l1_loss.item()
                    mean_ltec_loss += ltec_loss.item()
                    mean_cx_loss += cx_loss.item()
                    cnt += 1

                    images = [real_A, real_out, fake_out]
                    names = ['real_A', 'real', 'fake']

                    img_path = file_name
                    vutils.save_image(fake_out, os.path.join(web_dir, str(i).zfill(4) + '.png'), normalize=True)
                    vutils.save_image(real_out, os.path.join(web_dir, str(i).zfill(4) + 's.png'), normalize=True)
                webpage.save()
                mean_l1_loss /= cnt
                mean_ltec_loss /= cnt
                mean_cx_loss /= cnt
                with open(l1_loss_file, "a") as f:
                    f.write(
                        str(mean_l1_loss) + "\t" + "\t" + str(mean_cx_loss) + "\t" + "\t" + str(mean_ltec_loss) + "\n")
            ###############################################################################################

            # if epoch % 100 == 0:
            #     model.save_networks(opt.save_path)

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            print(time.time() - epoch_start_time)
            model.update_learning_rate()

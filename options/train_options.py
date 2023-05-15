from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.add_argument('--save_path', type=str, default='test_dse_songti01',
                            help='')
        parser.add_argument('--A_paths', type=str, default='heiti',
                            help='')
        parser.add_argument('--B_paths', type=str, default='songti',
                            help='')
        parser.add_argument('--CA_paths', type=str, default='heiti-01',
                            help='')
        parser.add_argument('--C_paths', type=str, default='songti-01',
                            help='')

        # lambda parameters
        parser.add_argument('--lambda_L1', type=float, default=150.0,
                            help='(agisnet) model weight for |C-G(A, E(Cs))|')
        parser.add_argument('--lambda_L1_B', type=float, default=0,
                            help='agisnet model weight for |B-gray(G(A, E(Cs)))|')
        parser.add_argument('--lambda_CX', type=float, default=25,
                            help='weight for contextual loss')
        parser.add_argument('--lambda_CX_B', type=float, default=0,
                            help='weight for contextual loss on B')
        parser.add_argument('--lambda_GAN', type=float, default=1,
                            help='weight on D loss. D(G(C, E(C)))')
        parser.add_argument('--lambda_GAN_B', type=float, default=0,
                            help='weight on D (for B) loss, D(G(B, E(B)))')
        parser.add_argument('--lambda_Ltec', type=float, default=100,
                            help='weight on D (for B) loss, D(G(B, E(B)))')
        parser.add_argument('--lambda_local_D', type=float, default=0,
                            help='weight on local adv loss')
        parser.add_argument('--cross', type=int, default=[0,0,0,0,0,0],
                           help='weight on local adv loss')

        parser.add_argument('--display_freq', type=int, default=1000,
                            help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=5,
                            help='if positive, display all images in a single visdom web' +
                            'panel with certain number of images per row.')
        parser.add_argument('--display_winsize', type=int,
                            default=256, help='display window size')
        parser.add_argument('--display_id', type=int, default=0,
                            help='window id of the web display')
        parser.add_argument('--display_port', type=int,
                            default=8097, help='visdom display port')
        parser.add_argument('--display_server', type=str,
                            default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--update_html_freq', type=int, default=4000,
                            help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=1000,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=10000,
                            help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--black_epoch_freq', type=int, default=0,
                            help='frequency of black epoch')
        parser.add_argument('--validate_freq', type=int, default=25,
                            help='frequency of log out all test result')
        parser.add_argument('--continue_train', action='store_true',
                            help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by ' +
                            '<epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str,
                            default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=200,
                            help='# of iter(epoch) at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100,
                            help='# of iter(epoch) to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float,
                            default=0.9, help='momentum term of adam')
        parser.add_argument('--no_html', action='store_true',
                            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # learning rate
        parser.add_argument('--lr', type=float, default=0.0002,
                            help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='lambda',
                            help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=100,
                            help='multiply by a gamma every lr_decay_iters iterations')


        parser.add_argument('--block_size', type=int, default=32,
                            help='block size of the image')
        parser.add_argument('--block_num', type=int, default=2,
                            help='block num of the image')

        # lambda for bicycle gan
        parser.add_argument('--lambda_GAN2', type=float, default=1.0, help='weight on D2 loss, D(G(A, random_z))')
        parser.add_argument('--lambda_z', type=float, default=0.5, help='weight for ||E(G(random_z)) - random_z||')
        parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for KL loss')
        parser.add_argument('--use_same_D', action='store_true', help='if two Ds share the weights or not')


        # for test LX
        ############################################################
        parser.add_argument('--results_dir', type=str,
                            default='test_result', help='saves results here.')
        parser.add_argument('--num_test', type=int, default=10000,
                            help='how many test images to run')
        parser.add_argument('--n_samples', type=int,
                            default=4, help='#samples, first one encode, others random')
        self.isTrain = True
        return parser

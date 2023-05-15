import os.path
import random

from PIL import Image, ImageFilter

from data.base_dataset import BaseDataset, transform_few_with_label
from data.image_folder import make_dataset

# /home/chenxu/comparasion/DSE3/data
class CnFewFusionDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def rreplace(self, s, old, new, occurrence):
        li = s.rsplit(old, occurrence)
        return new.join(li)

    def initialize(self, opt, k='train'):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_ABC = os.path.join(opt.dataroot)
        if k == 'train':
            self.A_paths = sorted(make_dataset(os.path.join(self.dir_ABC, 'train','glyph', opt.A_paths)))[:775]
            self.C_paths = sorted(make_dataset(os.path.join(self.dir_ABC, 'train', 'artistic', opt.C_paths)))[:775]
        else:
            self.A_paths = sorted(make_dataset(os.path.join(self.dir_ABC,'train', 'glyph', opt.A_paths)))[775:972]
            self.C_paths = sorted(make_dataset(os.path.join(self.dir_ABC, 'train', 'artistic', opt.C_paths)))[775:972]
        self.B_paths = sorted(make_dataset(os.path.join(self.dir_ABC, 'train', 'glyph', opt.B_paths)))[:775]
        self.CA_paths = sorted(make_dataset(os.path.join(self.dir_ABC, 'train', 'artistic', opt.CA_paths)))[:775]
        

        ############

    def __getitem__(self, index):


        A = Image.open(self.A_paths[index]).convert('RGB').resize([128, 128], Image.ANTIALIAS)
        B = Image.open(self.A_paths[index]).convert('RGB').resize([128, 128], Image.ANTIALIAS)
        C = Image.open(self.C_paths[index]).convert('RGB').resize([128, 128], Image.ANTIALIAS)
        Shapes = []
        Colors_CA = []
        Colors = []
        Style_paths = []


        label = 1.0
        # for shapes
        listB = random.sample(self.B_paths, self.opt.nencode*2)
        listCA = random.sample(self.CA_paths, self.opt.nencode)
        listC = random.sample(self.C_paths, self.opt.nencode)
        for i in range(self.opt.nencode):
            Colors_CA.append(Image.open(listCA[i]).convert('RGB').resize([128, 128], Image.ANTIALIAS))
            Colors.append(Image.open(listC[i]).convert('RGB').resize([128, 128], Image.ANTIALIAS))
        for i in range(self.opt.nencode):
            Shapes.append(Image.open(listB[i]).convert('RGB').resize([128, 128], Image.ANTIALIAS))

        A, B, C, label, Shapes, Colors, Colors_CA = \
            transform_few_with_label(self.opt, A, B, C, label, Shapes, Colors, Colors_CA)

        # A is the reference, B is the gray shape, C is the gradient
        return {'A': A, 'B': B, 'C': C, 'label': label,
                'Shapes': Shapes, 'Colors': Colors, 'Colors_CA': Colors_CA,
                'ABC_path': self.A_paths[index], 'Style_paths': Style_paths,
                }

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'CnFewFusionDataset'

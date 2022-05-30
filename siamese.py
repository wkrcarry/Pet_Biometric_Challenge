import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from nets.dennet_siamese import densenet121
from nets.resnet_siamese import resnet50
from nets.swin_siamese import swin_base_patch4_window7_224
from nets.vgg_siamese import vgg16

from utils.utils import letterbox_image, preprocess_input, cvtColor, show_config



class Siamese(object):
    _defaults = {

            model_path = "logs/dennet_last_epoch_weights.pth" ,
            '''
            model_path = "logs/resnet50_last_epoch_weights.pth"
            model_path = "logs/swin_last_epoch_weights.pth"
            model_path = "logs/vgg16_last_epoch_weights.pth"
            '''

        "input_shape"       : [224, 224],

        "letterbox_image"   : False,

        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.generate()
        
        show_config(**self._defaults)
        

    def generate(self):
        #   载入模型与权值
        print('Loading weights into state dict...')
        device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = densenet121()
        '''
        model = resnet50()
        model = swin_base_patch4_window7_224()
        model = vgg16()
        '''


        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
    
    def letterbox_image(self, image, size):
        image   = image.convert("RGB")
        iw, ih  = image.size
        w, h    = size
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image       = image.resize((nw,nh), Image.BICUBIC)
        new_image   = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        if self.input_shape[-1]==1:
            new_image = new_image.convert("L")
        return new_image
        

    def detect_image(self, image_1, image_2):

        image_1 = cvtColor(image_1)
        image_2 = cvtColor(image_2)
        

        image_1 = letterbox_image(image_1, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        image_2 = letterbox_image(image_2, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        

        photo_1  = preprocess_input(np.array(image_1, np.float32))
        photo_2  = preprocess_input(np.array(image_2, np.float32))

        with torch.no_grad():

            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(photo_1, (2, 0, 1)), 0)).type(torch.FloatTensor)
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(photo_2, (2, 0, 1)), 0)).type(torch.FloatTensor)
            
            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            #   获得预测结果，output输出为概率

            output = self.net([photo_1, photo_2])[0]
            output = torch.nn.Sigmoid()(output)

        return output

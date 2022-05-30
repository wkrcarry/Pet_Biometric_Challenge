import torch
import torch.nn as nn
from nets.vgg import VGG16
# class dense_block(nn.Module):
#     def __init__(self, in_channel, growth_rate, num_layers):
#         super(dense_block, self).__init__()
#         block = []
#         channel = in_channel
#         for i in range(num_layers):
#             block.append(conv_block(channel, growth_rate))
#             channel += growth_rate
#         self.net = nn.Sequential(*block)
#     def forward(self, x):
#         for layer in self.net:
#             out = layer(x)
#             x = torch.cat((out, x), dim=1)
#         return x
# def conv_block(in_channel, out_channel):
#     layer = nn.Sequential(
#         nn.BatchNorm2d(in_channel),
#         nn.ReLU(),
#         nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
#     )
#     return layer


# def transition(in_channel, out_channel):
#     trans_layer = nn.Sequential(
#         nn.BatchNorm2d(in_channel),
#         nn.ReLU(),
#         nn.Conv2d(in_channel, out_channel, 1),
#         nn.AvgPool2d(2, 2)
#     )
#     return trans_layer

# def _make_dense_block(self,channels, growth_rate, num):
#     block = []
#     block.append(dense_block(channels, growth_rate, num))
#     channels += num * growth_rate

#     return nn.Sequential(*block)
# def _make_transition_layer(self,channels):
#     block = []
#     block.append(transition(channels, channels // 2))
#     return nn.Sequential(*block)

# class densenet(nn.Module):
#     def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16]):
#         super(densenet, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(in_channel, 64, 7, 2, 3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.MaxPool2d(3, 2, padding=1)
#             )
#         self.DB1 = self._make_dense_block(64, growth_rate,num=block_layers[0])
#         self.TL1 = self._make_transition_layer(256)
#         self.DB2 = self._make_dense_block(128, growth_rate, num=block_layers[1])
#         self.TL2 = self._make_transition_layer(512)
#         self.DB3 = self._make_dense_block(256, growth_rate, num=block_layers[2])
#         self.TL3 = self._make_transition_layer(1024)
#         self.DB4 = self._make_dense_block(512, growth_rate, num=block_layers[3])
#         self.global_average = nn.Sequential(
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1,1)),
#         )
#         self.classifier = nn.Linear(1024, num_classes)
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.DB1(x)
#         x = self.TL1(x)
#         x = self.DB2(x)
#         x = self.TL2(x)
#         x = self.DB3(x)
#         x = self.TL3(x)
#         x = self.DB4(x)
#         x = self.global_average(x)
#         x = x.view(x.shape[0], -1)
#         x = self.classifier(x)
#         return x


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height) 
    
class vgg16(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()
        self.vgg = VGG16(pretrained, 3)
        del self.vgg.avgpool
        del self.vgg.classifier
        
        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        x1 = self.vgg.features(x1)
        x2 = self.vgg.features(x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x

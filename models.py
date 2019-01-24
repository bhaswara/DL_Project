import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                    nn.Conv2d(128, 128, 3, 1, 0),
                    nn.InstanceNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(128, 128, 3, 1, 0),
                    nn.InstanceNorm2d(128)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__()

        model = [nn.ReflectionPad2d(3),
        nn.Conv2d(input_nc, 32, 7, 1, 0),
        nn.InstanceNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, 2, 1),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 3, 2, 1),
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),
        ]


        for _ in range(6):
            model += [ResidualBlock()]


        model += [
        nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
        nn.InstanceNorm2d(32),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(3),
        nn.Conv2d(32, output_nc, 7, 1, 0),
        nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        model = [nn.Conv2d(input_nc, 64, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 4, 2, 1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, 4, 2, 1),
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256, 512, 4, 1, 1),
        nn.InstanceNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(512, 1, 4, 1, 1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x
        #return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)

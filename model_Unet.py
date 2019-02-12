import torch
import torch.nn as nn 
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        self.c0 = nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)
        self.c1 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.c2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.c3 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.c4 = nn.Conv2d(512, 512, 4, stride=2, padding=1)
        self.c5 = nn.Conv2d(512, 512, 4, stride=2, padding=1)
        self.c6 = nn.Conv2d(512, 512, 4, stride=2, padding=1)
        self.c7 = nn.Conv2d(512, 512, 4, stride=2, padding=1)

        self.d7 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        self.d6 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.d5 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.d4 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.d3 = nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1)
        self.d2 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        self.d1 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.d0 = nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1)
       
        self.bnc1 = nn.InstanceNorm2d(128)
        self.bnc2 = nn.InstanceNorm2d(256)
        self.bnc3 = nn.InstanceNorm2d(512)
        self.bnc4 = nn.InstanceNorm2d(512)
        self.bnc5 = nn.InstanceNorm2d(512)
        self.bnc6 = nn.InstanceNorm2d(512)

        self.bnd7 = nn.InstanceNorm2d(512)
        self.bnd6 = nn.InstanceNorm2d(512)
        self.bnd5 = nn.InstanceNorm2d(512)
        self.bnd4 = nn.InstanceNorm2d(512)
        self.bnd3 = nn.InstanceNorm2d(256)
        self.bnd2 = nn.InstanceNorm2d(128)
        self.bnd1 = nn.InstanceNorm2d(64)
        
    def forward(self, x):           
        en0 = self.c0(x)
        en1 = self.bnc1(self.c1(F.leaky_relu(en0, negative_slope=0.2)))
        en2 = self.bnc2(self.c2(F.leaky_relu(en1, negative_slope=0.2)))
        en3 = self.bnc3(self.c3(F.leaky_relu(en2, negative_slope=0.2)))
        en4 = self.bnc4(self.c4(F.leaky_relu(en3, negative_slope=0.2)))
        en5 = self.bnc5(self.c5(F.leaky_relu(en4, negative_slope=0.2)))
        en6 = self.bnc6(self.c6(F.leaky_relu(en5, negative_slope=0.2)))
        en7 = self.c7(F.leaky_relu(en6, negative_slope=0.2))

        de7 = self.bnd7(self.d7(F.relu(en7)))
        de6 = F.dropout(self.bnd6(self.d6(F.relu(torch.cat((en6, de7),1)))))
        de5 = F.dropout(self.bnd5(self.d5(F.relu(torch.cat((en5, de6),1)))))

        de4 = F.dropout(self.bnd4(self.d4(F.relu(torch.cat((en4, de5),1)))))
        de3 = self.bnd3(self.d3(F.relu(torch.cat((en3, de4),1))))
        de2 = self.bnd2(self.d2(F.relu(torch.cat((en2, de3),1))))
        de1 = self.bnd1(self.d1(F.relu(torch.cat((en1, de2),1))))

        de0 = F.tanh(self.d0(F.relu(torch.cat((en0, de1),1))))       

        return de0


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

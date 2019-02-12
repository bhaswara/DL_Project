import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

import argparse
import sys

from models import Generator
#from model_Unet import Generator

from utils import Loadimage

input_nc = 3
output_nc = 3

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='datasets/apple2orange/')
parser.add_argument('--opencv', action='store_true')
opt = parser.parse_args()



def test():
	if opt.opencv:
		print('OPENCV MODE')
	else:
		print('PIL MODE')

	G_AB = Generator(input_nc, output_nc)
	G_BA = Generator(output_nc, input_nc)
	
	G_AB.cuda()
	G_BA.cuda()

	G_AB.load_state_dict(torch.load('output/G_AB.pth'))
	G_BA.load_state_dict(torch.load('output/G_BA.pth'))

	G_AB.eval()
	G_BA.eval()

	Tensor = torch.cuda.FloatTensor 
	input_A = Tensor(1, 3, 256, 256)
	input_B = Tensor(1, 3, 256, 256)

	transforms_ = [transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

	dataloader = DataLoader(Loadimage(opt.dataroot,
		transforms_= transforms_, mode_train=False, mode_opencv=opt.opencv),
		batch_size=1, shuffle=False, num_workers=8)

	for i, batch in enumerate(dataloader):

		real_A = Variable(input_A.copy_(batch['A']))
		real_B = Variable(input_B.copy_(batch['B']))

		
		fake_B = 0.5*(G_AB(real_A).data + 1.0)
		fake_A = 0.5*(G_BA(real_B).data + 1.0)

		save_image(fake_A, 'output/A/%04d.png' % (i+1))
		save_image(fake_B, 'output/B/%04d.png' % (i+1))


		sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))


	sys.stdout.write('\n')	
	


if __name__ == '__main__':
	
	test()

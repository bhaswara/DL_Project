import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim 
import torch.nn as nn 

import itertools

from PIL import Image

import argparse
import transformcv as T

from models import Generator, Discriminator, weights_init
#from model_Unet import Generator, Discriminator, weights_init

from utils import Loadimage, keep, LambdaLR

import matplotlib.pyplot as plt 
import numpy as np 


input_nc = 3
output_nc = 3

n_epochs = 100
start_epoch = 0

lr_G = 0.0002
lr_D = 0.0002
decay = 50

loss_G_plot = []
loss_D_plot = []
loss_G_identity_plot = []
loss_G_GAN_plot = []
loss_G_cycle_plot = []



parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='datasets/apple2orange/')
parser.add_argument('--opencv', action='store_true')
opt = parser.parse_args()

def train():

	G_AB = Generator(input_nc, output_nc)
	G_BA = Generator(output_nc, input_nc)

	D_A = Discriminator(input_nc)
	D_B = Discriminator(output_nc)

	G_AB.cuda()
	G_BA.cuda()
	D_A.cuda()
	D_B.cuda()
	
	G_AB.apply(weights_init)
	G_BA.apply(weights_init)
	D_A.apply(weights_init)
	D_B.apply(weights_init)

	#Loss
	GD_loss = nn.MSELoss()
	L1_loss = nn.L1Loss()
	L1_loss_identity = nn.L1Loss()
	
	optim_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
		lr=lr_G, betas=(0.5, 0.999))
	optim_D_A = optim.Adam(D_A.parameters(), lr=lr_D, betas=(0.5, 0.999))
	optim_D_B = optim.Adam(D_B.parameters(), lr=lr_D, betas=(0.5, 0.999))

	lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, 
		lr_lambda=LambdaLR(n_epochs, start_epoch, decay).step)
	lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optim_D_A, 
		lr_lambda=LambdaLR(n_epochs, start_epoch, decay).step)
	lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optim_D_B, 
		lr_lambda=LambdaLR(n_epochs, start_epoch, decay).step)

	Tensor = torch.cuda.FloatTensor
	input_A = Tensor(1, 3, 256, 256)
	input_B = Tensor(1, 3, 256, 256)


	fake_A_buffer = keep()
	fake_B_buffer = keep()

	if opt.opencv:
		print('OPENCV MODE')
		transforms_ = [T.Scale(286),
		T.RandomCrop(256),
		T.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
	else:
		print('PIL MODE')
		transforms_ = [transforms.Resize(286, Image.BICUBIC),
			transforms.RandomCrop(256), 
    		transforms.RandomHorizontalFlip(),
    		transforms.ToTensor(),
    		transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]


	dataloader = DataLoader(Loadimage(opt.dataroot,
		transforms_=transforms_, unaligned=True, mode_opencv=opt.opencv),
		batch_size=1, shuffle=True, num_workers=8)

	for epoch in range(start_epoch, n_epochs):
		for i, batch in enumerate(dataloader):

			real_A = Variable(input_A.copy_(batch['A']))
			real_B = Variable(input_B.copy_(batch['B']))


			########################################
			#Train Generator
			#A to B
			optim_G.zero_grad()


			#identity loss
			same_B = G_AB(real_B)
			loss_identity_B = L1_loss_identity(same_B, real_B) * 5.0

			same_A = G_BA(real_A)
			loss_identity_A = L1_loss_identity(same_A, real_A) * 5.0

			fake_B = G_AB(real_A)
			pred_fake_B = D_B(fake_B)
			G_AB_Loss = GD_loss(pred_fake_B, 
				Variable(torch.ones(pred_fake_B.size()).cuda()))

			#B to A
			fake_A = G_BA(real_B)
			pred_fake_A = D_A(fake_A)
			G_BA_Loss = GD_loss(pred_fake_A, 
				Variable(torch.ones(pred_fake_A.size()).cuda()))


			#fake B to A
			similar_A = G_BA(fake_B)
			BA_cycle_loss = L1_loss(similar_A, real_A) * 10.0

			#fake A to B
			similar_B = G_AB(fake_A)
			AB_cycle_loss = L1_loss(similar_B, real_B) * 10.0

			#total loss G
			G_loss = G_AB_Loss + G_BA_Loss + BA_cycle_loss + AB_cycle_loss + loss_identity_A + loss_identity_B

			G_loss_identity = loss_identity_B + loss_identity_A
			G_loss_GAN = G_AB_Loss + G_BA_Loss
			G_loss_cycle = BA_cycle_loss + AB_cycle_loss

			#OptimizeG
			G_loss.backward()
			optim_G.step()

			loss_G_plot.append(G_loss.data[0])
			loss_G_identity_plot.append(G_loss_identity.data[0])
			loss_G_GAN_plot.append(G_loss_GAN.data[0])
			loss_G_cycle_plot.append(G_loss_cycle.data[0])

			#######################################
			#Train Discriminator
			#Discriminator D_AB
			optim_D_A.zero_grad()

			pred_real_A = D_A(real_A)
			D_real_loss = GD_loss(pred_real_A, 
				Variable(torch.ones(pred_real_A.size()).cuda()))
			fake_A = fake_A_buffer.empty_fill_data(fake_A)
			pred_d_fake_A = D_A(fake_A)
			D_fake_loss = GD_loss(pred_d_fake_A, 
				Variable(torch.zeros(pred_d_fake_A.size()).cuda()))

			D_A_loss_total = (D_real_loss + D_fake_loss) * 0.5
		
			D_A_loss_total.backward()
			optim_D_A.step()

			#Discriminator D_BA
			optim_D_B.zero_grad()

			pred_real_B = D_B(real_B)
			D_real_loss = GD_loss(pred_real_B, 
				Variable(torch.ones(pred_real_B.size()).cuda()))
			fake_B = fake_B_buffer.empty_fill_data(fake_B)
			pred_d_fake_B = D_B(fake_B)
			D_fake_loss = GD_loss(pred_d_fake_B, 
				Variable(torch.zeros(pred_d_fake_B.size()).cuda()))

			D_B_loss_total = (D_real_loss + D_fake_loss) * 0.5
			
			D_B_loss_total.backward()
			optim_D_B.step()

			D_Loss = D_A_loss_total + D_B_loss_total

			loss_D_plot.append(D_Loss.data[0])
			#####################################


			#Print All losses
			print('Epoch [%d/%d], Step [%d/%d], G_loss: %.4f, D_B_Loss: %.4f' 
				%(epoch +1, n_epochs, i+1, len(dataloader), G_loss.data[0], 
					D_Loss.data[0]))



		lr_scheduler_G.step()
		lr_scheduler_D_A.step()
		lr_scheduler_D_B.step()

		torch.save(G_AB.state_dict(), 'output/G_AB.pth')
		torch.save(G_BA.state_dict(), 'output/G_BA.pth')
		torch.save(D_A.state_dict(), 'output/D_AB.pth')
		torch.save(D_B.state_dict(), 'output/D_BA.pth')

	x = np.linspace(start_epoch, n_epochs, num=len(loss_G_plot))

	plt.figure(1)
	plt.plot(x, loss_G_plot)
	plt.xticks(np.arange(start_epoch,n_epochs+1, 25))
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Loss_G')
	plt.savefig('Loss_G.png')

	plt.figure(2)
	plt.plot(x, loss_G_identity_plot)
	plt.xticks(np.arange(start_epoch,n_epochs+1, 25))
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Loss_G_Identity')
	plt.savefig('Loss_G_Identity.png')

	plt.figure(3)
	plt.plot(x, loss_G_GAN_plot)
	plt.xticks(np.arange(start_epoch,n_epochs+1, 25))
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Loss_G_GAN')
	plt.savefig('Loss_G_GAN.png')

	plt.figure(4)
	plt.plot(x, loss_G_cycle_plot)
	plt.xticks(np.arange(start_epoch,n_epochs+1, 25))
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Loss_G_Cycle')
	plt.savefig('Loss_G_Cycle.png')

	plt.figure(5)
	plt.plot(x, loss_D_plot)
	plt.xticks(np.arange(start_epoch,n_epochs+1, 25))
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Loss_D')
	plt.savefig('Loss_D.png')



if __name__ == '__main__':

	train()





		
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import random
import os
from PIL import Image
from torch.autograd import Variable
import cv2

#Load Image
class Loadimage(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode_train=True, mode_opencv=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode_train = mode_train
        self.mode_opencv = mode_opencv

        if self.mode_train:
        	self.files_A = sorted(glob.glob(os.path.join(root, 'trainA/') + '*.*'))
        	self.files_B = sorted(glob.glob(os.path.join(root, 'trainB/') + '*.*'))
        else:
        	self.files_A = sorted(glob.glob(os.path.join(root, 'testA/') + '*.*'))
        	self.files_B = sorted(glob.glob(os.path.join(root, 'testB/') + '*.*'))

    def __getitem__(self, index):
    	if self.mode_opencv:
    		x_A = cv2.imread(self.files_A[index % len(self.files_A)])
    		x_A_cvt = cv2.cvtColor(x_A, cv2.COLOR_BGR2RGB)
    		item_A = self.transform(x_A_cvt)

    		if self.unaligned:
    			x_B = cv2.imread(self.files_B[random.randint(0, len(self.files_B) - 1)])
    			x_B_cvt = cv2.cvtColor(x_B, cv2.COLOR_BGR2RGB)
    			item_B = self.transform(x_B_cvt)
    		else:
    			x_B = cv2.imread(self.files_B[index % len(self.files_B)])
    			x_B_cvt = cv2.cvtColor(x_B, cv2.COLOR_BGR2RGB)
    			item_B = self.transform(x_B_cvt)
    	else:
    		item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
    		if self.unaligned:
    			item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
    		else:
    			item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

    	return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

		
class keep():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Check max size'
        self.max_size = max_size
        self.data = []

    def empty_fill_data(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
	"""docstring for LambdaLR"""
	def __init__(self, n_epoch, offset, decay_start_epoch):
		assert ((n_epoch - decay_start_epoch) > 0), "Decay must start before the training session ends!"
		self.n_epoch = n_epoch
		self.offset = offset
		self.decay_start_epoch = decay_start_epoch

	def step(self, epoch):
		return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epoch - self.decay_start_epoch)

		
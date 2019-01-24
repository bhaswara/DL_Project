import cv2
import numbers
import random

class Normalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std 

	def __call__(self, tensor):
		return (tensor - self.mean) / self.std


class Scale(object):
	def __init__(self, size, interpolation=cv2.INTER_CUBIC):
		self.size = size
		self.interpolation = interpolation

	def __call__(self, img):
		w, h = img.shape[1], img.shape[0]
		if (w <= h and w == self.size) or (h <= w and h == self.size):
			return img
		if w < h:
			ow = self.size
			oh = int(float(self.size) * h / w)
		else:
			oh = self.size
			ow = int(float(self.size) * w / h)

		return cv2.resize(img, dsize=(ow, oh), interpolation=self.interpolation)

class RandomCrop(object):
	def __init__(self, size):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size

	def __call__(self, img):
		w, h = img.shape[1], img.shape[0]
		th, tw = self.size
		if w == tw and h == th:
			return img

		x1 = random.randint(0, w - tw)
		y1 = random.randint(0, h - th)
		return img[y1:y1+th, x1:x1+tw]

class RandomHorizontalFlip(object):
	def __call__(self, img):
		if random.random() < 0.5:
			return cv2.flip(img, 1).reshape(img.shape)
		return img
		
		

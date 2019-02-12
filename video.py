import torch
from torch.autograd import Variable
from moviepy.editor import VideoFileClip


from models import Generator
#from model_Unet import Generator


import cv2
import numpy as np 
import argparse


input_nc = 3
output_nc = 3

parser = argparse.ArgumentParser()
parser.add_argument('--camera', action='store_true')
parser.add_argument('--record', action='store_true')
opt = parser.parse_args()

G_AB = Generator(input_nc, output_nc)
G_BA = Generator(output_nc, input_nc)


G_AB.cuda()
G_BA.cuda()

G_AB.load_state_dict(torch.load('output/G_AB.pth'))
G_BA.load_state_dict(torch.load('output/G_BA.pth'))


G_AB.eval()
G_BA.eval()


def camera():
	#fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
	#out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 360))
	cam = cv2.VideoCapture('01.mp4')
	#cam.set(3, 640)
	#cam.set(4, 480)

	while True:
		timer = cv2.getTickCount()
		ret_val, img = cam.read()

		img = cv2.resize(img, (640, 360))
 
		img = img.transpose((2,0,1))
		#img=torch.from_numpy(img).unsqueeze(0).float()
		img = np.expand_dims(img, axis=0)
		img = img/255.0
		img = torch.cuda.FloatTensor(img)
		img = img.cuda()
		img = Variable(img, volatile=True)

		fake_B = 0.5*(G_AB(img) + 1.0)

		fake_B_array = fake_B.data[0].cpu().numpy()
		fake_B_array = fake_B_array.transpose((1,2,0))

		fake_B_array = cv2.cvtColor(fake_B_array, cv2.COLOR_RGB2BGR)
		fake_B_array = cv2.resize(fake_B_array, (640, 480)) 
		fake_B_array_adjust = fake_B_array * 255

		fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)


		cv2.putText(fake_B_array, "FPS = %.2f" %fps, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
		cv2.putText(fake_B_array_adjust, "FPS = %.2f" %fps, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
		cv2.imshow('Tes', fake_B_array)
		#out.write(np.uint8(fake_B_array_adjust))


		key = cv2.waitKey(1)
		if key == 27:
			break
	
	cam.release()
	#out.release()
	cv2.destroyAllWindows()



def image_process():
	img = cv2.imread('check.jpg')

	h, w, _ = img.shape
	img = cv2.resize(img, (256, 256))
	img = img.transpose((2,0,1))
	img = np.expand_dims(img, axis=0)
	img = img/255.0
	img = torch.cuda.FloatTensor(img)
	img = img.cuda()
	img = Variable(img)

	fake_B = 0.5*(G_AB(img) + 1.0)

	fake_B_array = fake_B.data[0].cpu().numpy()
	fake_B_array = fake_B_array.transpose((1,2,0))

	fake_B_array = cv2.cvtColor(fake_B_array, cv2.COLOR_RGB2BGR)
	fake_B_array_adjust = fake_B_array * 255

	cv2.imshow('Tes', fake_B_array)
	cv2.imwrite('04.jpg', fake_B_array_adjust)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def read(image):
	#timer = cv2.getTickCount()
	#img = cv2.resize(image, (640, 480))
	img = np.array(image)
	img = img.transpose((2,0,1))
	img = np.expand_dims(img, axis=0)
	img = img/255.0
	img = torch.cuda.FloatTensor(img)
	img = img.cuda()
	img = Variable(img)

	fake_B = 0.5*(G_AB(img) + 1.0)
	fake_B_array = fake_B.data[0].cpu().numpy()
	fake_B_array = fake_B_array.transpose((1,2,0))
	fake_B_array = fake_B_array * 255
	image = fake_B_array.copy()

	#fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
	#cv2.putText(image, "FPS = %.2f" %fps, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


	return image

def record():
	vid_output = '01_x.mp4'

	clip1 = VideoFileClip('01.mp4')

	vid_clip = clip1.fl_image(read)
	vid_clip.write_videofile(vid_output, audio=False)



if __name__ == '__main__':
	if opt.camera:
		print('CAMERA MODE')
		camera()	
	elif opt.record:
		print('RECORD MODE')
		record()
	else:
		print('IMAGE MODE')
		image_process()

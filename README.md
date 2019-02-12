# DL_Project: Unpaired Day-to-Night translation using Cyclic GAN for Autonomous Driving

This is the github link for Deep Learning Project.

We used the same network as mention in the Cycle GAN paper. We also tried to change the Generator part with UNet. The images are shown as seen below.

<p align="center">
  <img width="600" height="300" src="https://user-images.githubusercontent.com/36017469/52666029-5cd42c80-2f0d-11e9-9ccb-334281254c9e.jpg">
</p>

<p align="center">
  <img width="600" height="300" src="https://user-images.githubusercontent.com/36017469/52666503-c0ab2500-2f0e-11e9-89d2-03275887d026.jpg">
</p>



For the python file, if you want to train the model just use: python main.py --dataroot datasets/bw2color/ --opencv . You can use PIL image by removing the '--opencv' command. This training will generate the trained data which is located inside /output folder.

To test the model, just run python test.py --dataroot/bw2color/ --opencv . This is same with PIL image if you train the model using PIL image. Just remove '--opencv' . This result will be saved inside /output folder

To process the video or another image just use video.py . There are 3 options there. If you use command '--camera', you can use your webcam or play video by changing VideoCapture(0) with VideoCapture('Your_VIDEO_NAME.mp4'). If you use command '--record', you will process a video and save it with new file name. And if you don't use any command, then it will be processed in image mode.
 

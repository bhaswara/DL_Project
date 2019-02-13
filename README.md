# DL_Project: Unpaired Day-to-Night translation using Cyclic GAN for Autonomous Driving

This is the github link for Deep Learning Project.

We used the same network as mention in the Cycle GAN paper. We also tried to change the Generator part with UNet. The images are shown below.

<p align="center">
  <img width="600" height="300" src="https://user-images.githubusercontent.com/36017469/52666029-5cd42c80-2f0d-11e9-9ccb-334281254c9e.jpg">
</p>

<p align="center">
  <img width="600" height="300" src="https://user-images.githubusercontent.com/36017469/52666503-c0ab2500-2f0e-11e9-89d2-03275887d026.jpg">
</p>


## Running
For the python file, if you want to train the model just use: 
```
python main.py --dataroot datasets/day2night --opencv
```
You can use PIL image format by removing the '--opencv' command. This training will generate the trained data which is located inside output/ folder.

To test the model, just run 
```
python test.py --dataroot datasets/day2night --opencv
```
This is same with PIL image if you train the model using PIL image. Just remove '--opencv' . This result will be saved inside output/A and output/B folder

To process the video or another image just use .
```
python video.py --COMMAND_MODE
```
There are 3 options there. If you use command '--camera', you can use your webcam or play video by changing VideoCapture(0) with VideoCapture('Your_VIDEO_NAME.mp4'). If you use command '--record', you will process a video and save it with new file name. And if you don't use any command, then it will process in image mode.


## Results
The following images are the results of the experiment that we mentioned on the paper. The youtube video and the plot of the results are also provided below.

<p align="center">
  <img width="500" height="1000" src="https://user-images.githubusercontent.com/36017469/52666929-d79e4700-2f0f-11e9-974d-ea72045d32ef.jpg">
</p>

<div align="center">
  <a href="https://www.youtube.com/watch?v=gzF1Z4PQaio"><img src="https://img.youtube.com/vi/gzF1Z4PQaio/0.jpg" alt="Cycle GAN Video"></a>
</div>

<p align="center">
  <img width="400" height="300" src="https://user-images.githubusercontent.com/36017469/52669669-f18f5800-2f16-11e9-93aa-3733a4faae8f.png">
</p>

<p align="center">
  <img width="400" height="300" src="https://user-images.githubusercontent.com/36017469/52669759-3f0bc500-2f17-11e9-9041-762af8b5929f.png">
</p>

<p align="center">
  <img width="400" height="300" src="https://user-images.githubusercontent.com/36017469/52669824-6c587300-2f17-11e9-99f3-5b72e787e0c6.png">
</p>

<p align="center">
  <img width="400" height="300" src="https://user-images.githubusercontent.com/36017469/52669890-97db5d80-2f17-11e9-8e70-e22767d77201.png">
</p>

<p align="center">
  <img width="400" height="300" src="https://user-images.githubusercontent.com/36017469/52669943-b5102c00-2f17-11e9-86da-bb60d7163a41.png">
</p>

# DL_Project

For the python file, if you want to train the model just use: python main.py --dataroot datasets/bw2color/ --opencv . You can use PIL image by removing the '--opencv' command. This training will generate the trained data which is located inside /output folder.

To test the model, just run python test.py --dataroot/bw2color/ --opencv . This is same with PIL image if you train the model using PIL image. Just remove '--opencv' . This result will be saved inside /output folder

To process the video or another image just use video.py . There are 3 options there. If you use command '--camera', you can use your webcam or play video by changing VideoCapture(0) with VideoCapture('Your_VIDEO_NAME.mp4'). If you use command '--record', you will process a video and save it with new file name. And if you don't use any command, then it will be processed in image mode.

For the training process, I haven't tried to plot the graph. Perhaps one of you can do it.

Thank you
 

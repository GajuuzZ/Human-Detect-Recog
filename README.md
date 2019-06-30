<h1> Human Tracking and Recognition Program. </h1>

Human detection tracking and recognition program via camera or video using Deep SORT, YOLOv3, and PCB.

### Dependencies

- Python > 3.6
- Pytorch > 0.3
- Tensorflow > 1.9.0

Need GPU to run smoothly.

### Basic Use

1. Train a Features Extraction model from 'Person_reID_baseline_pytorch-master' folder train PCB for basically.
<br> You may need to download [Market1501 Dataset](http://www.liangzheng.org/Project/project_reid.html) for training </br>
<br> More detail in *README.md*  </br>

2. Get pre-train YOLO model from : [yolo.h5](https://drive.google.com/file/d/1uvXFacPnrSMw6ldWTyLLjGLETlEsUvcE/view?usp=sharing)
and put it into Model folder.

3. Use *ExtractFromVid.py* to crop person image from video to Sample/ALL folder. Then make subfolder and name it for 
each person you want program to recognize. (for other person you don't want put them into one Unknow folder)

4. Use *MakeSampleSet.py* to make a sample set file and SVM model from all subfolder in *Sample*.

5. Check path and parameter before run *Main.py* file.

### Reference

PCB : https://github.com/layumi/Person_reID_baseline_pytorch
Deep SORT and YOLO : https://github.com/Qidian213/deep_sort_yolov3

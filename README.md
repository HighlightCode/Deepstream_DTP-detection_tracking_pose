# Deepstream_DTP-detection_tracking_pose

<img width="40%" src="https://user-images.githubusercontent.com/51946218/148735430-1c1452d7-0691-4f66-bb2f-d244ac15ad96.gif"/>



# Deepstream_DTP-detection_tracking_pose #
inspired by https://github.com/NVIDIA-AI-IOT/deepstream_pose_estimation.git project, i made a simple project to pose estimate single person with <b>Object Detection -> Object Tracking -> Pose Estimation process</b> . The large Difference from previous work is that i use the <b>detection -> tracking method (like updown method) </b> to detect the small object things. so as you can see in my project, the Tensor RT Engine detects the small object very well. the performance benchmark for my project is that i can get over >20fps. 



# Secondary Inference #
For Secondary Inference i used the model from https://github.com/YuliangXiu/MobilePose to inference pose estimation. but i changed the post processing network so without writing any Tensor RT postprocessing network, i can get a simple network outputs from my model then get pose data to numpy arrays. to see the network difference, see below.
<table>
  <tr>
    <td align=center><b>Before changing Network</b></td>
    <td align=center><b>After changing Network</b></td>
  </tr>
  <tr>
    <td align="center"><img width="40%" img src="https://user-images.githubusercontent.com/51946218/148738976-bf42de7f-63b1-4a1f-839b-bfc44ffd794d.jpg"></td>
    <td align="center"><img width="40%" img src="https://user-images.githubusercontent.com/51946218/148739178-432ac769-01fd-40c8-89e1-cc80832c7335.jpg"></td>
  </tr>
 </table>



# Prerequisites"
running environments
 1. DeepstreamSDK 6.0
 2. CUDA 10.2
 3. TensorRT 8.0.1
 4. Jetson nano



# Getting Started #
Before you get started, pls check the environment settings. 
- Install [DeepStream](https://developer.nvidia.com/deepstream-sdk) on your platform, verify it is working by running deepstream-app.
- check the TensorRT engine runs well on your platform

```
  $ https://github.com/Justdoit19/Deepstream_DTP-detection_tracking_pose.git
  $ check the config file path.
  $ python3 main.py -i <your video path> -s <store option>
```

- if you set the store option to <true>, then the output file will be saved on the ./result dir.
- i used two models to detect object, Yolo v5s, and caffe_resnet10 model by Nvidia and by experience, and caffe_resnet10 model which provided by Nvidia, is more accurate than the Yolov5s model. and be aware of the model can track only one object because of the performance issues on jetson nano.
  
NOTE: If you have another platform, or OS like Jetson Tegra, NX board , you should rechange the onnx file to TRT engine, since TRT engine has hardware dependencies. 
  
  
## TODO
- make more robust models.
- make robust tracking settings.
- increase performance to >60fps.
  
  
## References
  1. https://github.com/NVIDIA-AI-IOT/deepstream_pose_estimation
  2. https://github.com/YuliangXiu/MobilePose
  3. https://github.com/wang-xinyu/tensorrtx

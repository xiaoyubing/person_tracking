使用卡尔曼滤波和匈牙利算法进行的多人跟踪
----
- Author: xiaoyubin 
- Project: 人体跟踪
- Date: 2018年12月15日23:43:36


- Install [pytorch 0.4.0](https://github.com/pytorch/pytorch) and other dependencies.
  ```Shell
  pip install -r requirements.txt
  ```
- Download the models manually: **yolov3-spp.weights**([Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC) | [Baidu pan](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)). Place them into  `./models/yolo` respectively.


## Quick Start:  
$ python3 personTracking.py  

- Pre-requisite:  
    - torch 0.4.0
    - Python3.5  
    - Numpy  
    - SciPy  
    - Opencv 3.4.4 for Python

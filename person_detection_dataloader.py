#!/usr/bin/python
# -*- coding: UTF-8 -*-
from tracker import Tracker
import os
import torch
from torch.autograd import Variable
from fn import cropBox, im_to_torch
from opt import opt
from yolo.preprocess import prep_frame
from yolo.util import dynamic_write_results
from yolo.darknet import Darknet
import cv2
import numpy as np
import sys
import time
import torch.multiprocessing as mp
from multiprocessing import Queue as pQueue
from threading import Thread
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue, LifoQueue
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue, LifoQueue

class VideoLoader:
    def __init__(self, path, batchSize=1, queueSize=50):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.path = path

        (filepath, tempfilename) = os.path.split(self.path)
        (shotname, extension) = os.path.splitext(tempfilename)
        self.shotname = shotname  # 文件名
        # print(filepath,tempfilename,shotname,extension)

        self.stream = cv2.VideoCapture(path)
        assert self.stream.isOpened(), 'Cannot capture source'
        self.stopped = False
        

        self.batchSize = batchSize
        self.datalen = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)  # 帧率

        print('fps %s  总长为：%s', self.fps, self.datalen)

        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover

        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def length(self):
        return self.datalen

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        stream = cv2.VideoCapture(self.path)
        assert stream.isOpened(), 'Cannot capture source'

        for i in range(self.num_batches):
            img = []
            orig_img = []
            im_name = []
            im_dim_list = []
            for k in range(i*self.batchSize, min((i +  1)*self.batchSize, self.datalen)):
                inp_dim = int(opt.inp_dim)
                (grabbed, frame) = stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.Q.put((None, None, None, None))
                    print('===========================> This video get '+str(k)+' frames in total.')
                    sys.stdout.flush()
                    return
                # process and add the frame to the queue
                img_k, orig_img_k, im_dim_list_k = prep_frame(frame, inp_dim)
            
                img.append(img_k)
                orig_img.append(orig_img_k)
                im_name.append(str(k)+'.jpg')
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                # Human Detection
                img = torch.cat(img)
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

            while self.Q.full():
                time.sleep(2)
            
            self.Q.put((img, orig_img, im_name, im_dim_list, self.shotname, self.fps))

    def videoinfo(self):
        # indicate the video info
        fourcc=int(self.stream.get(cv2.CAP_PROP_FOURCC))
        fps=self.stream.get(cv2.CAP_PROP_FPS)
        frameSize=(int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return (fourcc,fps,frameSize)

    def getitem(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        return self.Q.qsize()


class DetectionLoader:
    def __init__(self, dataloder, batchSize=1, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()

        self.stopped = False
        self.dataloder = dataloder
        self.batchSize = batchSize
        self.datalen = self.dataloder.length()
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        # keep looping the whole dataset
        for i in range(self.num_batches):
            img, orig_img, im_name, im_dim_list, shotname, fps = self.dataloder.getitem()
            if img is None:
                self.Q.put((None, None, None, None, None, None, None, shotname, fps))
                return

            with torch.no_grad():
                # Human Detection
                img = img.cuda()
                prediction = self.det_model(img, CUDA=True)
                # NMS process
                dets = dynamic_write_results(prediction, opt.confidence,
                                    opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    for k in range(len(orig_img)):
                        if self.Q.full():
                            time.sleep(2)
                        self.Q.put((orig_img[k], im_name[k], None, None, None, None, None, shotname, fps))
                    continue
                dets = dets.cpu()
                im_dim_list = torch.index_select(im_dim_list,0, dets[:, 0].long())
                scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                
                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]

            for k in range(len(orig_img)):
                boxes_k = boxes[dets[:,0]==k]
                if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                    if self.Q.full():
                        time.sleep(2)
                    self.Q.put((orig_img[k], im_name[k], None, None, None, None, None, shotname, fps))
                    continue
                inps = torch.zeros(boxes_k.size(0), 3, opt.inputResH, opt.inputResW)
                pt1 = torch.zeros(boxes_k.size(0), 2)
                pt2 = torch.zeros(boxes_k.size(0), 2)
                if self.Q.full():
                    time.sleep(2)
                self.Q.put((orig_img[k], im_name[k], boxes_k, scores[dets[:,0]==k], inps, pt1, pt2, shotname, fps))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class DetectionProcessor:
    def __init__(self, detectionLoader, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.detectionLoader = detectionLoader
        self.stopped = False
        self.datalen = self.detectionLoader.datalen

        # initialize the queue used to store data
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = pQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        # keep looping the whole dataset
        for i in range(self.datalen):
            
            with torch.no_grad():
                (orig_img, im_name, boxes, scores, inps, pt1, pt2, shotname, fps) = self.detectionLoader.read()
                if orig_img is None:
                    self.Q.put((None, None, None, None, None, None, None, shotname, fps))
                    return
                if boxes is None or boxes.nelement() == 0:
                    while self.Q.full():
                        time.sleep(0.2)
                    self.Q.put((None, orig_img, im_name, boxes, scores, None, None, shotname, fps))
                    continue
                inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

                while self.Q.full():
                    time.sleep(0.2)
                self.Q.put((inps, orig_img, im_name, boxes, scores, pt1, pt2, shotname, fps))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


font = cv2.FONT_HERSHEY_SIMPLEX
track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]


class DataWriter:
    def __init__(self, save_video=False,
                savepath='examples/res/1.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=25, frameSize=(640,480),
                queueSize=1024,bg_color=None):
        """
        提取骨架信息，合成最后视频
        :param save_video: 是否保存合成骨架信息的视频
        :param savepath: 保存视频的路径
        :param fourcc: 保存视频的格式
        :param fps: 保存视频的帧率
        :param frameSize: 保存视频的尺寸
        :param queueSize:
        :param bg_color:保存视频的背景，默认None使用原视频作为背景，white用纯白作为背景，black用黑色作为背景
        """
        if save_video:
            # initialize the file video stream along with the boolean
            # used to indicate if the thread should be stopped or not
            self.stream = cv2.VideoWriter(savepath, fourcc, fps, frameSize)
            assert self.stream.isOpened(), 'Cannot open video for writing'
        self.save_video = save_video
        self.stopped = False
        self.final_result = []
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')
        self.bg_color = bg_color
        self.current_frame = 0
        self.current_sec = 0
        self.out_img_dir = '0_3'
        self.tracker = Tracker(160, 30, 50, 100)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                if self.save_video:
                    self.stream.release()
                return
            # otherwise, ensure the queue is not empty
            if not self.Q.empty():
                (boxes, scores, hm_data, pt1, pt2, orig_img, im_name, shotname, fps) = self.Q.get()
                orig_img = np.array(orig_img, dtype=np.uint8)
                if boxes is None:
                    print('boxes~~~~~~~~~~~~~~~~~~~~~is None')
                    img = orig_img
                    if opt.vis:
                        cv2.imshow("AlphaPose Demo", img)
                        cv2.waitKey(30)
                else:
                    centers = []
                    img = orig_img
                    pedestrians = {}
                    counter = 0
                    for i, box in enumerate(boxes):
                        cv2.putText(img, "i:%s" % i, (box[0], box[1] + 20), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                        # center_point = self.center(box)
                        min_x = min(box[0].int(), box[2].int())
                        max_x = max(box[0].int(), box[2].int())
                        min_y = min(box[1].int(), box[3].int())
                        max_y = max(box[1].int(), box[3].int())

                        x = (min_x + max_x) / 2
                        y = (min_y + max_y) / 2
                        w = max_x - min_x
                        h = max_y - min_y

                        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                        # cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
                        b = np.array([[x], [y]])
                        centers.append(np.round(b))
                        counter += 1

                    # Track object using Kalman Filter
                    # 使用卡尔曼滤波器跟踪对象
                    self.tracker.Update(centers)

                    # For identified object tracks draw tracking line
                    # 对于已识别的目标轨迹，绘制跟踪线
                    # Use various colors to indicate different track_id
                    # 使用不同的颜色表示不同的轨迹ID
                    for i in range(len(self.tracker.tracks)):
                        if (len(self.tracker.tracks[i].trace) > 1):
                            for j in range(len(self.tracker.tracks[i].trace) - 1):
                                # Draw trace line
                                x1 = self.tracker.tracks[i].trace[j][0][0]
                                y1 = self.tracker.tracks[i].trace[j][1][0]
                                x2 = self.tracker.tracks[i].trace[j + 1][0][0]
                                y2 = self.tracker.tracks[i].trace[j + 1][1][0]
                                clr = self.tracker.tracks[i].track_id % 9
                                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                                         track_colors[clr], 2)

                    # Display the resulting tracking frame
                    cv2.imshow('Tracking', img)

                    # Slower the FPS
                    cv2.waitKey(50)
            else:
                time.sleep(0.1)

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        return not self.Q.empty()

    def save(self, boxes, scores, hm_data, pt1, pt2, orig_img, im_name, shotname, fps):
        # save next frame in the queue
        self.Q.put((boxes, scores, hm_data, pt1, pt2, orig_img, im_name, shotname, fps))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        time.sleep(0.2)

    def results(self):
        # return final result
        return self.final_result

    def len(self):
        # return queue len
        return self.Q.qsize()


def crop_from_dets(img, boxes, inps, pt1, pt2):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]
        if width > 100:
            scaleRate = 0.2
        else:
            scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inps[i] = cropBox(tmp_img, upLeft, bottomRight, opt.inputResH, opt.inputResW)
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2

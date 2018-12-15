#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
    File name         : tracker.py
    File Description  : 使用卡尔曼滤波和匈牙利算法进行多目标跟踪
    Author            : 
    Date created      : 2018年12月14日18:21:38
    Date last modified: 2018年12月14日18:21:49
    Python Version    : 3.5
"""

import numpy as np
from kalman_filter import KalmanFilter
from common import dprint
from scipy.optimize import linear_sum_assignment


class Track(object):
    """为要跟踪的每个对象创建跟踪类
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount):
        """初始化Track类使用的变量
        Args:
            prediction: predicted centroids of object to be tracked预测被跟踪对象的几何中心
            trackIdCount: identification of each track object每个被跟踪对象的标识
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = KalmanFilter()  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path


class Tracker(object):
    """Tracker class that updates track vectors of object tracked更新被跟踪对象的跟踪向量的跟踪器类
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for the track object undetected
                                允许跳过未检测到的跟踪对象的最大帧数
            max_trace_lenght: trace path history length 跟踪路径历史长度
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.trackIdCount = trackIdCount
        self.tracks = []

    def Update(self, detections):
        """Update tracks vector using following steps:
            使用以下步骤更新跟踪向量
            - Create tracks if no tracks vector found
            创建跟踪器，如果没有找到跟踪器载体
            - Calculate cost using sum of square distance between predicted vs detected centroids
            利用预测的质心与检测到的质心之间的平方和计算成本
            - Using Hungarian Algorithm assign the correct detected measurements to predicted tracks
            使用匈牙利算法将正确的检测值分配给预测轨迹
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            识别没有任务的轨迹，如果有的话
            - If tracks are not detected for long time, remove them
            如果长时间没有检测到跟踪，请删除它们
            - Now look for un_assigned detects
            现在查找没有分配的检测
            - Start new tracks
            开始新的跟踪
            - Update KalmanFilter state, lastResults and tracks trace
            更新卡尔曼滤波器的状态，最后的结果和跟踪器的轨迹
        Args:
            detections: detected centroids of object to be tracked 检测到要跟踪的对象的质心
        Return:
            None
        """

        # Create tracks if no tracks vector found
        # 创建跟踪器，如果没有找到跟踪器载体
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between predicted vs detected centroids
        # 利用预测的质心与检测到的质心之间的平方和计算成本
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix 成本矩阵
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = self.tracks[i].prediction - detections[j]
                    distance = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])
                    cost[i][j] = distance
                except:
                    pass

        # Let's average the squared ERROR 求平方误差的平均值
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements to predicted tracks
        # 使用匈牙利算法将正确的检测值分配给预测轨迹
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        # 识别没有分配的轨迹，如果有的话
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # 检查成本距离阈值
                # If cost is very high then un_assign (delete) the track
                # 如果成本非常高，那么不分配(删除)轨迹
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        # 如果长时间没有检测到跟踪，请删除它们
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max 只有当跳过的帧超过最大
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    dprint("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        # 现在查找没有分配的检测
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],
                              self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            detections[assignment[i]], 1)
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            np.array([[0], [0]]), 0)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction

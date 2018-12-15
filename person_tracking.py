#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import copy
from detectors import Detectors
from tracker import Tracker


def main():
    """Main function for multi person tracking
    Usage:
        $ python3.5 person_tracking.py
    Pre-requisite:
        - Python3.5
        - Numpy
        - SciPy
        - Opencv 3.4.4 for Python
    Args:
        None
    Return:
        None
    """

    # Create opencv video capture object
    cap = cv2.VideoCapture('10.avi')

    # Create Object Detector
    detector = Detectors()

    # Create Object Tracker
    # tracker = Tracker(160, 30, 5, 100)
    tracker = Tracker(160, 30, 50, 100)

    # Variables initialization
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False

    # Infinite loop to process video frames
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Make copy of original frame
        orig_frame = copy.copy(frame)

        # Skip initial frames that display logo
        if (skip_frame_count < 15):
            skip_frame_count += 1
            continue

        # Detect and return centeroids of the objects in the frame
        # 检测并返回帧中对象的中心
        # TODO 该检测点可以个跟换成yolo v3人体检测
        centers = detector.Detect(frame)

        # If centroids are detected then track them
        # 如果检测到质心，就跟踪它们
        if (len(centers) > 0):

            # Track object using Kalman Filter
            # 使用卡尔曼滤波器跟踪对象
            tracker.Update(centers)

            # For identified object tracks draw tracking line
            # 对于已识别的目标轨迹，绘制跟踪线
            # Use various colors to indicate different track_id
            # 使用不同的颜色表示不同的轨迹ID
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j+1][0][0]
                        y2 = tracker.tracks[i].trace[j+1][1][0]
                        clr = tracker.tracks[i].track_id % 9
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 2)

            # Display the resulting tracking frame
            cv2.imshow('Tracking', frame)

        # Display the original frame
        cv2.imshow('Original', orig_frame)

        # Slower the FPS
        cv2.waitKey(50)

        # Check for key strokes
        k = cv2.waitKey(50) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break
        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            pause = not pause
            if (pause is True):
                print("Code is paused. Press 'p' to resume..")
                while (pause is True):
                    # stay in this loop until
                    key = cv2.waitKey(30) & 0xff
                    if key == 112:
                        pause = False
                        print("Resume code..!!")
                        break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute main
    main()

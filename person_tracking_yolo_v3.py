#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch.utils.data
import numpy as np
from opt import opt
from person_detection_dataloader import VideoLoader, DetectionLoader, DetectionProcessor, DataWriter
import os
import sys
from tqdm import tqdm
from fn import getTime
import cv2


args = opt
args.dataset = 'coco'
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    src_video_path = args.video
    mode = args.mode
    bg_color = args.bgcolor
    outputpath = args.outputpath
    # if not os.path.exists(args.outputpath):
    #     os.mkdir(args.outputpath)

    if not len(src_video_path):
        raise IOError('Error: must contain --video')

    ext = src_video_path[-3:]
    if ext.lower() not in ['mp4', 'avi', 'mpg']:
        print('%s文件不是视频文件跳过', src_video_path)
        exit(0)

    print("视频文件，进行骨架提取", src_video_path)
    file_name = os.path.split(src_video_path)[-1][:-4]

    # Load input video
    data_loader = VideoLoader(src_video_path, batchSize=args.detbatch).start()
    (fourcc,fps,frameSize) = data_loader.videoinfo()

    # Load detection loader
    print('Loading YOLO model..')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
    det_processor = DetectionProcessor(det_loader).start()
    
    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Data writer
    save_path = os.path.join(outputpath, file_name + '.avi')
    # save_path = os.path.join(args.outputpath, 'AlphaPose_'+ntpath.basename(videofile).split('.')[0]+'.avi')
    writer = DataWriter(False, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize, bg_color=bg_color).start()

    im_names_desc =  tqdm(range(data_loader.length()))
    batchSize = args.posebatch
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2, shotname, fps) = det_processor.read()
            if orig_img is None:
                break
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1], shotname, fps)
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)
            # Pose Estimation
            
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            # for j in range(num_batches):
            #     inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
            #     hm_j = pose_model(inps_j)
            #     hm.append(hm_j)
            # hm = torch.cat(hm)
            # ckpt_time, pose_time = getTime(ckpt_time)
            # runtime_profile['pt'].append(pose_time)
            #
            # hm = hm.cpu().data
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1], shotname, fps)

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        if args.profile:
            # TQDM
            im_names_desc.set_description(
            'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
            )

    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    while(writer.running()):
        pass
    writer.stop()
    # final_result = writer.results()
    # write_json(final_result, args.outputpath)

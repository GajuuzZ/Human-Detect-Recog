#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:08:38 2019

@author: gjz
"""
import os
import sys
import cv2
import numpy as np
import argparse
import time
import pickle
import yaml
from collections import deque
from shapely.geometry import Polygon

from yolo import YOLO
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from model import PCB, torch

CONFIG_FILE = 'Model/PCB01.yaml'  # Path to Features Model yaml file.
WEIGHT_FILE = 'Model/PCB01.pth'  # Path to Features Model pth file.
RECOG_MODEL = 'Model/sample_set1-svm.pkl'  # Path to SVM model file.
RECOG_SAMPLE = 'Sample/sample_set1.pkl'  # Path to sample set file.


def reduceFeaturesSample(ft, lb, n):
    cm = nn_matching._cosine_distance(ft, ft[:1])
    clss = list(set(lb))
    r = np.linspace(0, 1, n)
    new_ft = []
    new_lb = []
    for c in clss:
        idx = np.where(np.array(lb) == c)[0]
        ft_c = ft[idx]
        cm_c = cm[idx]
        for j in range(len(r) - 1):
            idx = np.where(np.logical_and(cm_c > r[j], cm_c <= r[j + 1]))[0]
            if len(idx) == 0:
                continue
            ftt = ft_c[idx]
            ftt = np.mean(ftt, 0)
            new_ft.append(ftt)
            new_lb.append(c)

    return np.array(new_ft), new_lb


def getColr(name):
    if name == 'Unknown':
        color = (150, 150, 150)
    else:
        color = (0, 0, 255)
    return color


def cosine_match(sample, query, threshold=0.5):
    cm = nn_matching._cosine_distance(sample, query)
    if cm.min() < threshold:
        idx = np.where(cm == cm.min())[0]
        name = lb_set[idx[0]]
    else:
        name = 'Unknown'
    return name, 1.0 - cm.min()


def svm_match(svm, query):
    name = svm.predict_proba(query)
    return cls_names[np.argmax(name[0])], max(name[0])


def selectPolygon(sel_image):
    selectPolygon.done = False
    selectPolygon.current = (0, 0)
    selectPolygon.points = []
    window_name = 'selectROI'

    def on_mouse(event, x, y, flags, param):
        if selectPolygon.done:
            return

        if event == cv2.EVENT_MOUSEMOVE:
            selectPolygon.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            selectPolygon.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            selectPolygon.done = True

    cv2.imshow(window_name, sel_image)
    cv2.waitKey(1)
    cv2.setMouseCallback(window_name, on_mouse)

    while not selectPolygon.done:
        canvas = np.copy(sel_image)
        if len(selectPolygon.points) > 0:
            cv2.polylines(canvas, np.array([selectPolygon.points]),
                          False, (0, 255, 0), 1)
            cv2.line(canvas, selectPolygon.points[-1],
                     selectPolygon.current, (255, 255, 255))
        cv2.imshow(window_name, canvas)
        if cv2.waitKey(50) == 27:
            selectPolygon.done = True

    canvas = np.copy(sel_image)
    if len(selectPolygon.points) > 0:
        cv2.drawContours(canvas, np.array([selectPolygon.points]), 0,
                         (0, 255, 0))
    print('ESC to continue.')
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    return selectPolygon.points


def iou_poly(poly1, poly2):
    p1 = Polygon(poly1)
    p2 = Polygon(poly2)
    area = p1.intersection(p2).area
    if area > 0.0:
        area = area / p1.area
    return area


def bb2poly(bbx):
    return [(bbx[0], bbx[1]), (bbx[2], bbx[1]), (bbx[2], bbx[3]), (bbx[0], bbx[3])]


def calculate_dest(point_list):
    dx = point_list[0][0] - point_list[2][0]
    dy = point_list[0][1] - point_list[2][1]
    m = dy / dx if dx != 0 else 0
    b = point_list[0][1] - (m * point_list[0][0])
    d = np.abs(max(point_list[0][0] - point_list[-1][0],
                   point_list[0][1] - point_list[-1][1]))

    if np.sign(dx) == 1:
        x = point_list[0][0] + d
        y = (x * m) + b
    else:
        x = point_list[0][0] - d
        y = (x * m) + b

    return int(x), int(y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human Detection Recognition Program.')
    parser.add_argument('-C', '--camera',
                        default='./Video/TownCentreXVID.avi',  # Path to video or camera address.
                        help='Camera or Video to open.')
    parser.add_argument('-R', '--resize', type=str, default='640x480',
                        help='If provided, Resize image before process. Recommend : 640x480.')
    parser.add_argument('-S', '--stealth', default=False,
                        help='Stealth mode: 0/1')
    parser.add_argument('--SVM', default=False,
                        help='Use SVM as features recognition matchs.')
    parser.add_argument('--save_vid_name', type=str, default='')
    parser.add_argument('--draw_result', default=True)
    parser.add_argument('--roi', type=str, default=None,
                        help='Region of interest file. If not exist will select one.')

    args = parser.parse_args()

    # DeepSORT Parameters.
    max_cosine_distance = 0.3
    nn_budget = 20
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=30)

    # Others.
    buffer = 6  # Green line direction buffer.
    alarm_cooldown = 15  # second.

    # Yolo model for object detection.
    print('Load Detection Model.')
    print('Use YOLO model for Detection.')
    yolo = YOLO()

    # Human feature extraction.
    print('Load Feature Model.')
    print('Use PCB model for features extractor.')
    with open(CONFIG_FILE, 'r') as stream:
        config = yaml.load(stream)
    nclasses = config['nclasses']
    encoder = PCB(nclasses)
    encoder.load_state_dict(torch.load(WEIGHT_FILE))
    encoder = encoder.eval().cuda()

    print('Load Recognition Model.')
    if args.SVM:
        print('Use SVM for matching.')
        recog = pickle.load(open(RECOG_MODEL, 'rb'))  # SVM match.
        _, _, cls_names = pickle.load(open(RECOG_SAMPLE, 'rb'))
    else:
        print('Use Cosine for matching.')
        ft_set, lb_set, cls_names = pickle.load(open(RECOG_SAMPLE, 'rb'))
        # ft_set, lb_set = reduceFeaturesSample(np.array(ft_set), lb_set, 20)

    spl = args.resize.split('x')
    w = int(spl[0])
    h = int(spl[1])

    out_vid = False
    if args.save_vid_name != '':
        out_vid = True
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(args.save_vid_name, fourcc, 15, (w, h))

    print('Start Video Stream.')
    cap = cv2.VideoCapture(args.camera)
    fps_time = 0

    if args.roi is not None:
        print('Select Region.')
        ret, frame = cap.read()
        if not os.path.exists(args.roi):
            if args.camera == 0:
                frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (w, h), cv2.INTER_AREA)
            roi = selectPolygon(frame)
            with open(args.roi, 'wb') as f:
                pickle.dump(roi, f)
        else:
            with open(args.roi, 'rb') as f:
                roi = pickle.load(f)
        last_alarm = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                if args.camera == 0:
                    frame = cv2.flip(frame, 1)
                if out_vid and not args.draw_result:
                    image = np.copy(frame)

                frame = cv2.resize(frame, (w, h), cv2.INTER_AREA)
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                alarm = False

                # Detect person.
                boxs, box_clss, scrs = yolo.predict(frame, filters=['person'])
                if len(boxs) > 0:
                    boxs = boxs[np.where(scrs > 0.5)[0]]  # Make sure that is a human.

                # Get features from each person.
                features = encoder.encode(frame, boxs)

                # Get pair of detection and features.
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
                # Use only nearly full body detected.
                detections = [d for d in detections if d.to_xyah()[2] < 0.55]

                # Do tracking.
                tracker.predict()
                tracker.update(detections)

                id_found = []
                for i, track in enumerate(tracker.tracks):
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()

                    center = (int(bbox[0] + ((bbox[2] - bbox[0]) / 2)),
                              int(bbox[1] + ((bbox[3] - bbox[1]) / 2)))

                    # Add Track trail.
                    if not hasattr(track, 'trail'):
                        track.trail = deque(maxlen=buffer)
                    track.trail.appendleft(center)

                    # Calculate Destination of Track.
                    if len(track.trail) > 3:
                        track.dest = calculate_dest(track.trail)
                    else:
                        track.dest = center
                    track.dest_bbx = (int(track.dest[0] - ((bbox[2] - bbox[0]) / 2)),
                                      int(track.dest[1] - ((bbox[3] - bbox[1]) / 2)),
                                      int(track.dest[0] + ((bbox[2] - bbox[0]) / 2)),
                                      int(track.dest[1] + ((bbox[3] - bbox[1]) / 2)))

                    # Identify person.
                    if args.SVM:
                        pred, confident = svm_match(recog, np.array(track.features))
                    else:
                        pred, confident = cosine_match(ft_set, np.array(track.features))

                    # Make sure the Identify.
                    if not hasattr(track, 'scan'):
                        track.scan = deque(maxlen=5)
                        track.conf = deque(maxlen=5)
                        track.clss = 'Unknown'
                        track.confident = 0
                    track.scan.appendleft(pred)
                    track.conf.appendleft(confident)
                    if len(track.scan) == 5 and track.clss == 'Unknown' and \
                            len(set(track.scan)) == 1:
                        track.clss = pred
                        track.confident = np.mean(track.conf)

                    # Set Alarm.
                    tpass = False  # Trespassing into area.
                    if args.roi is not None:
                        # Use IOU to set alarm.
                        if track.clss != 'Guard':
                            iou = iou_poly(bb2poly(track.dest_bbx), roi)
                            if iou > 0.45 and track.confident > 0:
                                alarm = True
                                tpass = True

                    # Draw Result.
                    if not args.stealth:
                        clr1 = getColr(track.clss)
                        clr2 = getColr(pred)
                        clr3 = (255, 255, 255)

                        if tpass:
                            clr3 = (0, 0, 255)

                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                                      (int(bbox[2]), int(bbox[3])),
                                      clr1, 2)
                        cv2.putText(frame, '{0}, {1}:{2:.2f}'.format(track.track_id,
                                                                     track.clss, track.confident),
                                    (int(bbox[0]) - 10, int(bbox[1]) - 20),
                                    0, 5e-3 * 100, clr1, 2)
                        cv2.putText(frame, '{0}:{1:.2f}'.format(pred, confident),
                                    (int(bbox[0]), int(bbox[1]) - 10), 0, 5e-3 * 60,
                                    clr2, 1)

                        for t in np.arange(1, len(track.trail)):
                            thickness = int(np.sqrt(buffer / float(t + 1)) * 2.5)
                            cv2.line(frame, track.trail[t - 1], track.trail[t],
                                     (0, 255, 0), thickness)

                        cv2.line(frame, track.trail[0], track.dest, clr3, 2)
                        cv2.rectangle(frame, (track.dest_bbx[0], track.dest_bbx[1]),
                                      (track.dest_bbx[2], track.dest_bbx[3]),
                                      clr3, 2)

                # Alarming.
                if args.roi is not None and alarm:
                    if time.time() - last_alarm >= alarm_cooldown:
                        print('Alarm')
                        last_alarm = time.time()

                fps = (1.0 / (time.time() - fps_time))
                sys.stdout.write('\r' + 'FPS: %.2f' % fps)
                sys.stdout.flush()

                # Display video.
                if not args.stealth:
                    if args.roi is not None:
                        cv2.drawContours(frame, np.array([roi]), 0, (0, 255, 0))
                    cv2.imshow('frame', frame)

                # Write output video file.
                if out_vid:
                    if not args.draw_result:
                        out.write(image)
                    else:
                        out.write(frame)

                fps_time = time.time()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                print('Can not read video camera!')
                break

    except KeyboardInterrupt:
        print('End Program!')
        pass

    if out_vid:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

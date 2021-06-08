import os
import time
import cv2 as cv

net = cv.dnn_DetectionModel('frozen_inference_graph.pb', 'graph.pbtxt')

net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

net.setInputSize(300, 300)
net.setInputSwapRB(True)

path = '<YOUR PATH HERE>'

images = [k for k in os.listdir(path) if '.jpg' in k]

for image in images:
    frame = cv.imread(os.path.join(path, image))
    start_time = time.time()
    classes, confidences, boxes = net.detect(frame, confThreshold=0.60)
    elapsed_time = time.time() - start_time
    print('time: {:.3f}ms'.format(elapsed_time * 1000))
    if not len(boxes) == 0:
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            print(classId, confidence)
            cv.rectangle(frame, box, color=(0, 255, 0), thickness=2)
    cv.imshow('out', frame)
    key = cv.waitKey(0) & 0xFF
    if key == ord('q'):
        break

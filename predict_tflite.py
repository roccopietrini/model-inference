import argparse
import time
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import cv2


def prepare_input(image_path):
	""" Input image preprocessing for SSD MobileNet format
	args:
		image_path: path to image
	returns:
		input_data: numpy array of shape (1, width, height, channel) after preprocessing
	"""
	# NxHxWxC, H:1, W:2
	height = input_details[0]['shape'][1]
	width = input_details[0]['shape'][2]
	orig_img = Image.open(image_path)
	img = orig_img.convert('RGB').resize((width, height))

	# add N dim
	input_data = np.expand_dims(img, axis=0)

	return (2.0/255.0)*input_data.astype(np.float32)-1.0, orig_img

def postprocess_output(img_width, img_height):
	""" Output post processing
	args:
		image_path: path to image
	returns:
		boxes: numpy array (num_det, 4) of boundary boxes at image scale
		classes: numpy array (num_det) of class index
		scores: numpy array (num_det) of scores
		num_det: (int) the number of detections
	"""
	# SSD Mobilenet tflite model returns 10 boxes by default.
	# Use the output tensor at 4th index to get the number of valid boxes
	num_det = int(interpreter.get_tensor(output_details[3]['index']))
	boxes = interpreter.get_tensor(output_details[0]['index'])[0][:num_det]
	classes = interpreter.get_tensor(output_details[1]['index'])[0][:num_det]
	scores = interpreter.get_tensor(output_details[2]['index'])[0][:num_det]

	# Scale the output to the input image size
	#img_width, img_height = Image.open(image_path).size # PIL


	df = pd.DataFrame(boxes)
	df['ymin'] = df[0].apply(lambda y: max(1,(y*img_height)))
	df['xmin'] = df[1].apply(lambda x: max(1,(x*img_width)))
	df['ymax'] = df[2].apply(lambda y: min(img_height,(y*img_height)))
	df['xmax'] = df[3].apply(lambda x: min(img_width,(x * img_width)))
	boxes_scaled = df[['ymin', 'xmin', 'ymax', 'xmax']].values
	return boxes_scaled, classes, scores, num_det

def draw_boundaryboxes(orig_image):
	""" Draw the detection boundary boxes
	args:
		image_path: path to image
	"""
	# Draw detection boundary boxes
	dt_boxes, dt_classes, dt_scores, num_det = postprocess_output(orig_image.width, orig_image.height)
	opencv_img = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
	for i in range(num_det):
		if int(dt_scores[i]*100) > 60:
			print(dt_scores[i]*100)
			[ymin, xmin, ymax, xmax] = list(map(int, dt_boxes[i]))
			cv2.rectangle(opencv_img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
			cv2.putText(opencv_img, '{}% score'.format(int(dt_scores[i]*100)), (xmin, ymin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10,255,0), 1)
	return opencv_img


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-i',
		'--image',
		default='.',
		help='image path for object detection')
	parser.add_argument(
		'-m',
		'--model_file',
		default='ssd_mobilenet_oid_v1_float.tflite',
		help='.tflite model to be executed')

	args = parser.parse_args()
	path = args.image

	images = [k for k in os.listdir(path) if '.jpg' in k]

	interpreter = tf.lite.Interpreter(model_path=args.model_file)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	for image in images:
		input_data, orig_image = prepare_input(os.path.join(path, image))
		interpreter.set_tensor(input_details[0]['index'], input_data)
		start_time = time.time()
		interpreter.invoke()
		stop_time = time.time()
		print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
		frame = draw_boundaryboxes(orig_image)
		cv2.imshow('out', frame)
		key = cv2.waitKey(0) & 0xFF
		if key == ord('q'):
			break

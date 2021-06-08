# model-inference
Collection of scripts for inference of a SSD Detection Model trained with Tensorflow Object Detection API. Starting from a SSD-like object detection model, like MobileNetV3-SSD, we have different options to do inference. I collected here three of them, benchmarking on a general purpouse CPU and a Raspberry Pi4. All the scripts expect a folder with test images and will iterate over the images, pressing any key in the prediction window. The scripts for inference are named predict<..>, be sure to edit them before run with the required fields (take a look at the code and you'll notice). Part of this code comes from the official repository of Tensorflow.


## Requirements
* Model checkpoints and configuration file (pipeline.config) from the training. You can use a [model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) from the Tensorflow Object Detection API Model zoo to test this code.
* Tensorflow 1.15
* Opencv > 4.3
* Tensorflow Object Detection API
* Numpy
* Pillow
* Pandas

## Inference with pure Tensorfow
Starting point is to have checkpoints from a training job, we'll have 3 files like this: model.ckpt.data-00000-of-00001, model.ckpt.index, model.ckpt.meta. First we need to freeze the model for inference:

```bash
python3 export_inference_graph.py --pipeline_config_path=pipeline.config --trained_checkpoint_prefix model.ckpt --output_directory=. --add_postprocessing_op=true
```

This will generate a frozen_inference_graph.pb which can be used for inference:

```bash
python3 predict_tensorflow.py
```

## Inference with OpenCV
We can use the frozen_inference_graph already generated, together with a text-based representation of the model as described [here](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API). First, let's generate the text representation:

```bash
python3 tf_text_graph_ssd.py --input frozen_inference_graph.pb --config pipeline.config --output graph.pbtxt
```

We can now run:

```bash
python3 predict_opencv.py
```

## Inference with Tensorflow Lite
We use now export_tflite_ssd_graph.py script to take this original TF graph, and convert it into an 'intermediate' form that removes all the control-flow logic, and leaves the graph with raw outputs that can be directly fed to a custom TFLite op (thats specifically designed for NMS).
Then, you use tflite_convert to convert this intermediate representation into the final TFLite graph.

```bash
python3 export_tflite_ssd_graph.py --pipeline_config_path=pipeline.config --trained_checkpoint_prefix model.ckpt --output_directory=. --add_postprocessing_op=true
```

```bash
python3 tflite_converter.py
```

Now we have the final form of the Tensorflow Lite model `tflite_quant_graph.tflite`, ready for inference on device.

```bash
python3 predict_tflite.py --model tflite_quant_graph.tflite --image <PATH WITH IMAGES>
```

## Comparison

The main goal of this repo was to have a rough idea of the performances of different methods for inference on device. As you can see from the table, OpenCV outperforms the other methods (at least with this kind of models) with the advantage of a very easy to use API for detection models. On the other hand, OpenCV uses an independent engine and not everything that comes with tensorflow is yet supported, you have to check with the model you want to use. Approx average inference time in milliseconds is reported.
Model used is MobileNetv3-SSD from TFODAPI Model Zoo, retrained for the only class "person", on COCO 2017 dataset (selecting images with class person). For this test I took 5 random images from COCO 2017 containing class person. 
CPU used for this test is Intel i7-5500U CPU @ 2.40GHz.

Inference | Raspberry Pi4 | Desktop CPU
------------ | ------------ | -------------
Tensorflow | N/A | 600
OpenCV DNN Module | 95 | 26
Tensorflow Lite | 100 | 30

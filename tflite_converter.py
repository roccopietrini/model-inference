import tensorflow as tf

graph_def_file = 'tflite_graph.pb'
output_tflite_graph = 'tflite_quant_graph.tflite'
input_arrays = ['normalized_input_image_tensor']
output_arrays = ['TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2', 'TFLite_Detection_PostProcess:3']
input_shapes = {'normalized_input_image_tensor': [1, 300, 300, 3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays, input_shapes)
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
#converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
#converter.quantized_input_stats = {'normalized_input_image_tensor': (128, 128)}
tflite_model = converter.convert()
with open(output_tflite_graph, "wb") as tflite_file: 
    tflite_file.write(tflite_model)

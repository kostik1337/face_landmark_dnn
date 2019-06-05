import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.models import load_model

import sys
sys.path.append("../testing/")
from utils import relu6, smoothL1

test_models = False

def convert(modelPath):
    with custom_object_scope({'smoothL1': smoothL1, 'relu6': relu6}):
        converter = tf.lite.TFLiteConverter.from_keras_model_file(modelPath)
        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)
        if test_models: keras_model = load_model(modelPath)

    if test_models:
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the TensorFlow Lite model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]['index'])

        # Test the TensorFlow model on random input data.
        tf_results = keras_model.predict_on_batch(tf.constant(input_data))

        # Compare the result.
        for tf_result, tflite_result in zip(tf_results, tflite_results):
            np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)

if __name__ == '__main__':
    convert(sys.argv[1])
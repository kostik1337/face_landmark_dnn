import tensorflow as tf

from tensorflow.keras.utils import custom_object_scope

import sys
sys.path.append("../testing/")
from utils import relu6, smoothL1

def convert(modelPath):
    with custom_object_scope({'smoothL1': smoothL1, 'relu6': relu6}):
        converter = tf.lite.TFLiteConverter.from_keras_model_file(modelPath)
        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)

if __name__ == '__main__':
    convert(sys.argv[1])
import tensorflow as tf 
import pathlib

def squash_and_save_model(model):
    generator = model.generator
    generator.trainable = False
    for layer in generator.layers:
        layer.trainable = False
    
    g_converter = tf.lite.TFLiteConverter.from_keras_model(generator)
    g_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    g_tflite_quant_model  = g_converter.convert()

    tflite_models_dir = pathlib.Path("mocap_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_gen_model_file = tflite_models_dir/"generator.tflite"

    tflite_gen_model_file.write_bytes(g_tflite_quant_model)

    return g_tflite_quant_model

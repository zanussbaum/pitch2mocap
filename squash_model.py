import tensorflow as tf 
import pathlib

def squash_and_save_model(model):
    discriminator = model.discriminator
    generator = model.generator
    
    d_converter, g_converter = tf.lite.TFLiteConverter.from_keras_model(discriminator), tf.lite.TFLiteConverter.from_keras_model(generator)
    d_converter.optimizations, g_converter.optimizations = [tf.lite.Optimize.DEFAULT], [tf.lite.Optimize.DEFAULT]
    
    d_tflite_quant_model, g_tflite_quant_model  = d_converter.convert(), g_converter.convert()

    tflite_models_dir = pathlib.Path("mocap_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_gen_model_file = tflite_models_dir/"generator.tflite"
    tflite_discr_model_file = tflite_models_dir/"discriminator.tflite"

    tflite_discr_model_file.write_bytes(d_tflite_quant_model)
    tflite_gen_model_file.write_bytes(g_tflite_quant_model)

    return d_tflite_quant_model, g_tflite_quant_model

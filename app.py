import tensorflow as tf
import gradio as gr
import numpy as np
model = tf.keras.models.load_model('Model.h5')
def predict(inp):
  prediction = model.predict(np.array([tf.keras.preprocessing.image.img_to_array(inp)]) )
  return (1-prediction)*100
gr.Interface(fn=predict,inputs=gr.Image(shape=(224, 224)),outputs=gr.Number(),server_name="0.0.0.0").launch()

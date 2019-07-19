import logging
import numpy as np
from PIL import Image
import argparse
from tensorflow.python.platform import gfile
import cv2
import sys
import tensorflow as tf


def get_logger():

  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
  logger.addHandler(stdout_handler)
  return logger

logger = get_logger()

INPUT_SIZE = 513

class SegmentationModel(object):

  """Class to load the model and run the inference."""

  """The following fields are specific to DeepLabV3 and ought to be edited 
  in the event of running any other custom trained models."""
  
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

  def __init__(self, model_name):
  
    """Creates and loads the pretrained model."""
    self.graph = tf.Graph()

    graph_def = None
	
    """Extract the model from the frozen graph."""
	
    with gfile.FastGFile(model_name,'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

 
    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)
	
    logger.info("Model extracted successfully.")
	
  def run(self, image):
  
    """Runs inference on a single image.
    Args:
      image: A PIL.Image object, raw input image.
    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
  
    [width, height] = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def run_inference(original_im, MODEL):
  
  resized_im, seg_map = MODEL.run(original_im)
  width, height = original_im.size
  
  resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
  target_size = (int(resize_ratio * width), int(resize_ratio * height))
  resized_image = original_im.convert('RGB').resize(target_size, Image.ANTIALIAS)
  
  out_img_array = np.array(resized_image)
 
  rows, cols, channel = out_img_array.shape
  for x in range(0, rows):
    for y in range(0, cols):
        if(seg_map[x,y] != 15):
          out_img_array[x,y,:] = 255
  
  im = Image.fromarray(out_img_array.astype('uint8'), 'RGB')

  return im


def detect_and_save(input, output, MODEL):
  
  """Video capture."""
  
  vcapture = cv2.VideoCapture(input)
  width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = vcapture.get(cv2.CAP_PROP_FPS)

  """Define codec and create video writer."""
  
  file_name = output + ".avi"
  vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (513, 288))
  success = True
  count = 0
        
  while success:

    success, image = vcapture.read()    
     
    if success:
	
      """OpenCV returns images as BGR, we convert to RGB."""
	  
      image = image[..., ::-1]
        
      image = Image.fromarray(image)   
	  
      r = run_inference(image, MODEL)
            
      s = np.array(r)
                
      """RGB -> BGR to save image to video."""
	  
      splash = s[..., ::-1]
          
      """Add image to video writer."""
	  
      vwriter.write(splash)
	  
      count += 1
        
  vcapture.release()    
  vwriter.release()
  logger.info("Saved to " + file_name)
        

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('-i', action='store', required=True, dest='input_video', help='The input video for the background removal.')
  parser.add_argument('-o', action='store', required=True, dest='output_video', help='The file storing the output video post the background removal.')
  parser.add_argument('-m', action='store', required=True, dest='model_name', help='The file name of the model to be used in the segmentation process.')
  
  args = parser.parse_args()
  
  input = args.input_video
  output = args.output_video
  model_name = args.model_name
  
  del args
  
  MODEL = SegmentationModel(model_name)
  
  detect_and_save(input, output, MODEL) 

      
if __name__ == "__main__":
  
  main()

 


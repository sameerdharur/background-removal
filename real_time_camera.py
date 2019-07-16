import sys
import argparse
import logging
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image
import cv2
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
    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def run_visualization(MODEL):

  """Inferences DeepLab model and visualizes result."""
  
  cv2.namedWindow('Background Removal')
  camera = cv2.VideoCapture(0)
  
  while True:
    original_im = camera.read()[1]
    original_im = Image.fromarray(original_im, 'RGB')
    INPUT_SIZE = 513
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
    resized_output = Image.fromarray(out_img_array, 'RGB')
    resized_output = resized_output.resize((1600, 900))
    out_img_array = np.array(resized_output)  
    cv2.imshow('Background Removal',out_img_array)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
  camera.release()
  cv2.destroyAllWindows()


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('-m', action='store', required=True, dest='model_name', help='The file name of the model to be used in the segmentation process.')
  
  args = parser.parse_args()
  
  model_name = args.model_name
  
  del args

  MODEL = SegmentationModel(model_name)
  run_visualization(MODEL)

 
if __name__ == "__main__":

  main()
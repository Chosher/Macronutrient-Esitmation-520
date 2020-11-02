# coding=utf8
# This code is adapted from the https://github.com/tensorflow/models/tree/master/official/r1/resnet.
# ==========================================================================================
# NAVER’s modifications are Copyright 2020 NAVER corp. All rights reserved.
# ==========================================================================================
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import tensorflow as tf

from official.utils.export import export
from utils import data_util
from functions import data_config
import numpy as np
from tqdm import tqdm
from utils import log_utils
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.core.framework import graph_pb2

def export_test(bin_export_path, flags_obj, ir_eval):
  ds = tf.data.Dataset.list_files(flags_obj.data_dir + '/' + flags_obj.val_regex)
  ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=10)

  def parse_tfr(example_proto):
    feature_def = {'label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
                   'image': tf.FixedLenFeature([], dtype=tf.string, default_value='')}
    features = tf.io.parse_single_example(serialized=example_proto, features=feature_def)
    return features['image'], features['label']
  
  ds = ds.map(parse_tfr)   
  ds = ds.batch(flags_obj.val_batch_size)
  iterator = ds.make_one_shot_iterator()
  images, labels = iterator.get_next()
  dconf = data_config.get_config(flags_obj.dataset_name)
  num_val_images = dconf.num_images['validation']
  if flags_obj.zeroshot_eval or ir_eval:
    feature_dim = flags_obj.embedding_size if flags_obj.embedding_size > 0 else flags_obj.num_features
    np_features = np.zeros((num_val_images, feature_dim), dtype=np.float32)
    np_labels = np.zeros(num_val_images, dtype=np.int64)
    np_i = 0
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      tf.saved_model.load(sess=sess, export_dir=bin_export_path, tags={"serve"})
      for _ in tqdm(range(int(num_val_images / flags_obj.val_batch_size) + 1)):
        try:
          np_image, np_label = sess.run([images, labels])
          print(images)      
          print(np_image)
          
          np_predict = sess.run('embedding_tensor:0',
                                feed_dict={'input_tensor:0': np_image})
          np_features[np_i:np_i + np_predict.shape[0], :] = np_predict
          np_labels[np_i:np_i + np_label.shape[0]] = np_label
          np_i += np_predict.shape[0]

        except tf.errors.OutOfRangeError:
          break
      assert np_i == num_val_images

    from sklearn.preprocessing import normalize

    x = normalize(np_features)
    np_sim = x.dot(x.T)
    np.fill_diagonal(np_sim, -10)  # removing similarity for query.
    num_correct = 0
    for i in range(num_val_images):
      cur_label = np_labels[i]
      rank1_label = np_labels[np.argmax(np_sim[i, :])]
      if rank1_label == cur_label:
        num_correct += 1
    recall_at_1 = num_correct / num_val_images
    metric = recall_at_1
  else:
    np_i = 0
    correct_cnt = 0
    t = []
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      tf.saved_model.load(sess=sess, export_dir=bin_export_path, tags={"serve"})
      tf.logging.info(flags_obj.val_batch_size)
      for _ in tqdm(range(int(num_val_images / flags_obj.val_batch_size) + 1)):
        try:
          np_image, np_label = sess.run([images, labels])
          tf.logging.info(np_image[0])  
          break
          np_predict = sess.run('ArgMax:0',
                                feed_dict={'input_tensor:0': np_image})
          np_i += np_predict.shape[0]
          correct_cnt += np.sum(np_predict == np_label) 
        except tf.errors.OutOfRangeError:
          break
      assert np_i == num_val_images
      metric = correct_cnt / np_i
  return metric

def image_bytes_serving_input_fn(image_shape, decoder_name, dtype=tf.float32, pptype='imagenet'):
  """Serving input fn for raw jpeg images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    # Bounding box around the whole image.
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=dtype, shape=[1, 1, 4])
    _, _, num_channels = image_shape
    tf.logging.info("!!!!!!!!!! Preprocessing type for exporting pb: {} and decoder type: {}".format(pptype, decoder_name))
    image = data_util.preprocess_image(
      image_buffer=image_bytes, is_training=False, bbox=bbox,
      num_channels=num_channels, dtype=dtype, use_random_crop=False,
      decoder_name=decoder_name, dct_method='INTEGER_ACCURATE', preprocessing_type=pptype)
    return image

  image_bytes_list = tf.placeholder(
    shape=[None], dtype=tf.string, name='input_tensor')
  images = tf.map_fn(
    _preprocess_image, image_bytes_list, back_prop=False, dtype=dtype)
  return tf.estimator.export.TensorServingInputReceiver(
    images, {'image_bytes': image_bytes_list})


def export_pb(flags_core, flags_obj, shape, classifier, ir_eval=False):
  export_dtype = flags_core.get_tf_dtype(flags_obj)

  if not flags_obj.data_format:
    raise ValueError('The `data_format` must be specified: channels_first or channels_last ')

  bin_export_path = os.path.join(flags_obj.export_dir, flags_obj.data_format, 'binary_input')
  bin_input_receiver_fn = functools.partial(image_bytes_serving_input_fn, shape, flags_obj.export_decoder_type,
                                            dtype=export_dtype, pptype=flags_obj.preprocessing_type)

  pp_export_path = os.path.join(flags_obj.export_dir, flags_obj.data_format, 'preprocessed_input')
  pp_input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
    shape, batch_size=None, dtype=export_dtype)

  result_bin_export_path = classifier.export_savedmodel(bin_export_path, bin_input_receiver_fn)
  classifier.export_savedmodel(pp_export_path, pp_input_receiver_fn)

  if flags_obj.export_decoder_type == 'jpeg':
    metric = export_test(result_bin_export_path, flags_obj, ir_eval)
    msg = 'IMPOTANT! Evaluation metric of exported saved_model.pb is {}'.format(metric)
    tf.logging.info(msg)
    with tf.gfile.Open(result_bin_export_path.decode("utf-8") + '/model_performance.txt', 'w') as fp:
      fp.write(msg)
    
    
def _resize_image(image, height, width):
  """Simple wrapper around tf.resize_images.

  This is primarily to make sure we use the same `ResizeMethod` and other
  details each time.

  Args:
    image: A 3-D image `Tensor`.
    height: The target height for the resized image.
    width: The target width for the resized image.

  Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [height, width].
  """
  return tf.image.resize_images(
    image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
    align_corners=False)
    
def run_prediction(bin_export_path, img_path, save=False):  
  import pandas as pd
  #Macornutrient data
  macro_df = pd.read_csv('../FOODX-251_Dataset/macro_data_foodX251.csv')  
  label_df = pd.read_csv('../FOODX-251_Dataset/class_list.csv')  
  ds = tf.data.Dataset.list_files(img_path+'*/*')

  def parse_image(img):
    return img

  def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = parse_image(img)
    return img
  
  ds = ds.map(process_path)  
  ds = ds.batch(1)
  iterator = ds.make_one_shot_iterator()
  images = iterator.get_next()

  num_images = len(tf.io.gfile.listdir(img_path))
  np_i = 0
  correct_cnt = 0
  pred = []
  
  with tf.Session() as sess:
    #Load model    
    graph = tf.compat.v1.saved_model.loader.load(sess=sess, export_dir=bin_export_path, tags={"serve"})
    graph = graph.graph_def
    graph = tf.get_default_graph()
    #Get softmax output
    x = graph.get_tensor_by_name("softmax_tensor:0")
    
    #Create tensors to multiply output from softmax by the corresponding macronutrient data
    add_energy = tf.math.multiply(x,tf.convert_to_tensor(macro_df['Energy'].to_list(), dtype =tf.float32, name='e_con'), name='e_mul')
    add_fat = tf.math.multiply(x,tf.convert_to_tensor(macro_df['Fat'].to_list(), dtype =tf.float32 , name='f_con'), name='f_mul')
    add_protein = tf.math.multiply(x,tf.convert_to_tensor(macro_df['Protein'].to_list(),dtype =tf.float32, name='p_con'), name='p_mul')
    add_carb = tf.math.multiply(x,tf.convert_to_tensor(macro_df['Carbohydrate'].to_list(), dtype =tf.float32, name='c_con'), name='c_mul')
    
    #Sum the multiplcations--This is the output
    energy_output = (tf.math.reduce_sum(add_energy, name='energy_output'))
    fat_output = (tf.math.reduce_sum(add_fat, name='fat_output'))
    protein_output = (tf.math.reduce_sum(add_protein, name='protein_output' ))
    carb_output = (tf.math.reduce_sum(add_carb, name='carb_output'))
    
    
    #Get input
    input_tensor = graph.get_tensor_by_name("input_tensor:0")
    #Also get the chosen food
    output_tensor = graph.get_tensor_by_name("ArgMax:0")
    
    #Outputs for the model
    model_input = tf.saved_model.utils.build_tensor_info(input_tensor)
    model_output = tf.saved_model.utils.build_tensor_info(output_tensor)
    model_output1 = tf.saved_model.utils.build_tensor_info(energy_output)
    model_output2 = tf.saved_model.utils.build_tensor_info(fat_output)
    model_output3 = tf.saved_model.utils.build_tensor_info(protein_output)
    model_output4 = tf.saved_model.utils.build_tensor_info(carb_output)
    
    #Loop through each image. 
    for _ in tqdm(range(num_images)):
      try:
        np_image = sess.run(images)

        #Make predictions
        np_predict_Arg, np_predict_energy, np_predict_fat, np_predict_protein, np_predict_carb = sess.run(['ArgMax:0', 'energy_output:0', 'fat_output:0', 'protein_output:0', 'carb_output:0'], 
                              feed_dict={'input_tensor:0': np_image})
        #Get output
        pred.append([label_df['food'][np_predict_Arg[0]], np_predict_Arg, np_predict_energy, np_predict_fat, np_predict_protein, np_predict_carb])
      
      except tf.errors.OutOfRangeError:
        break
    if(save):
      save_model(graph, bin_export_path, img_path)
#   Return dataframe of predictions for images. 
    df = pd.DataFrame(pred, columns = ['Predicted food','Food Index', 'Energy', 'Fat', 'Protein', 'Carbohydrate'])  
    return df

def save_model(graph, bin_export_path, img_path):
  input_graph_def = graph.as_graph_def()
  minimal_graph = tf.graph_util.convert_variables_to_constants(sess,  input_graph_def, ['ArgMax', 'energy_output', 'fat_output', 'protein_output', 'carb_output'])

  tf.train.write_graph(minimal_graph, 'test', 'model', as_text=False)
    
    # Instantiate a SavedModelBuilder
    # Note that the serve directory is REQUIRED to have a model version subdirectory
  builder = tf.saved_model.builder.SavedModelBuilder("serve/1")

   # Read in ProtoBuf file
  with tf.gfile.GFile("test/model", "rb") as protobuf_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(protobuf_file.read())

  # Get input and output tensors from GraphDef
  # These are our injected bitstring layers
  [inp, out, out2, out3, out4, out5] = tf.import_graph_def(graph_def, name="", return_elements=["input_tensor:0", "ArgMax:0", "energy_output:0","fat_output:0","protein_output:0","carb_output:0"])
    
  out = tf.expand_dims(out, 0)
  out2 = tf.expand_dims(out2, 0)
  out3 = tf.expand_dims(out3, 0)
  out4 = tf.expand_dims(out4, 0)
  out5 = tf.expand_dims(out5, 0)

        # Build prototypes of input and output bitstrings
  input_bytes = tf.saved_model.utils.build_tensor_info(inp)
  output_bytes = tf.saved_model.utils.build_tensor_info(out)
  output_bytes2 = tf.saved_model.utils.build_tensor_info(out2)
  output_bytes3 = tf.saved_model.utils.build_tensor_info(out3)
  output_bytes4 = tf.saved_model.utils.build_tensor_info(out4)
  output_bytes5 = tf.saved_model.utils.build_tensor_info(out5)
   

        # Create signature for prediction
  signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"input": input_bytes},
        outputs={"output": output_bytes, "Energy": output_bytes2 ,"Fat": output_bytes3, "Protein": output_bytes4, "Carb": output_bytes5},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        # Add meta-information
  builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.
            DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition
        })

# Create the SavedModel
  builder.save()


#loads the graph into memory. 
def load_graph(model_filepath):
    print('Loading model...')
    with tf.Session() as sess:

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())


        # Define input tensor
        input1 = tf.placeholder(tf.uint8, shape = [None, 224, 224, 3], name='input_tensor')

        tf.import_graph_def(graph_def, {'input_tensor': input1})
        graph = tf.get_default_graph()

        print('Model loading complete!')
        
        for n in tf.get_default_graph().as_graph_def().node:
           if n.name == 'import_1/global_step':
             tf.logging.info(n)
        
        input_tensor = graph.get_tensor_by_name("input_tensor:0")
        output_tensor = graph.get_tensor_by_name("import_1/ArgMax:0")
        energy_output = graph.get_tensor_by_name("import_1/energy_output:0")
        protein_output = graph.get_tensor_by_name("import_1/protein_output:0")
        fat_output = graph.get_tensor_by_name("import_1/fat_output:0")
        carb_output = graph.get_tensor_by_name("import_1/carb_output:0")

        model_input = tf.saved_model.utils.build_tensor_info(input_tensor)
        model_output = tf.saved_model.utils.build_tensor_info(output_tensor)
        model_output1 = tf.saved_model.utils.build_tensor_info(energy_output)
        model_output2 = tf.saved_model.utils.build_tensor_info(fat_output)
        model_output3 = tf.saved_model.utils.build_tensor_info(protein_output)
        model_output4 = tf.saved_model.utils.build_tensor_info(carb_output)

        # build signature definition
        signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input_tensor': model_input},
        outputs={'output': model_output, 'energy': model_output1, 'fat': model_output2, 'protein': model_output3, 'carb': model_output4},
        method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder = tf.saved_model.builder.SavedModelBuilder('test/serve')

        builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition
        })
    # Save the model so we can serve it with a model server :)
        builder.save()
        

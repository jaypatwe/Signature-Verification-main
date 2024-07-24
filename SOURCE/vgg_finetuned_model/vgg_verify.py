import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import optimizers
import os
import cv2

def make_square(path):
    ''' Reize the image to 256x256 dimension '''
    image = cv2.imread(path)
    image = cv2.resize(image, (256, 256))
    cv2.imwrite('media/image.png', image)

def load_image(image_path):
    ''' Return the image in the format required by VGG16 model. '''
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_features(feature_extractor, image):
    ''' Returns the features extracted by the model. '''
    return feature_extractor.predict(load_image(image))

def cosine_similarity_fn(anchor_image_feature, test_image_feature):
    ''' Returns the features extracted by the model. '''
    return cosine_similarity(anchor_image_feature, test_image_feature)[0][0]


def verify(anchor_image, gan_op):
    # loads the model and removes the last layer is removed
    #vgg_model = tf.keras.models.load_model('saved_model.pb') #Depracated in keras3 ValueError: File format not supported: filepath=saved_model.pb. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(saved_model.pb, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).
    #vgg_model = tf.keras.models.load_model(r"C:\Users\aksha\Desktop\Git Cloned Repositories\Signature-Verification_System_using_YOLOv5-and-CycleGAN\Streamlit_App\SOURCE\vgg_finetuned_model\SavedModel")
    #vgg_model = tf.keras.layers.TFSMLayer(r"C:\Users\aksha\Desktop\Git Cloned Repositories\Signature-Verification_System_using_YOLOv5-and-CycleGAN\Streamlit_App\SOURCE\vgg_finetuned_model\SavedModel", call_endpoint='serving_default') #Source https://keras.io/api/layers/backend_specific_layers/tfsm_layer/
    #vgg_model = tf.keras.models.load_model(r"C:\Users\aksha\Desktop\Git Cloned Repositories\Signature-Verification_System_using_YOLOv5-and-CycleGAN\Streamlit_App\SOURCE\vgg_finetuned_model\SavedModel")
    #vgg_model = tf.saved_model.load(r"C:\Users\aksha\Desktop\Git Cloned Repositories\Signature-Verification_System_using_YOLOv5-and-CycleGAN\Streamlit_App\SOURCE\vgg_finetuned_model\SavedModel")
    #feature_extractor.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),
    #          metrics=['accuracy'])
    #feature_extractor = tf.keras.Sequential(vgg_model.layers[:-1])



    # WORKING MODELS
    feature_extractor = tf.keras.models.load_model(r"C:\Users\aksha\Desktop\Git Cloned Repositories\Signature-Verification_System_using_YOLOv5-and-CycleGAN\Streamlit_App\SOURCE\vgg_finetuned_model\bbox_regression_cnn.h5")
    ####### feature_extractor = tf.keras.models.load_model(r"C:\Users\aksha\Desktop\Git Cloned Repositories\Signature-Verification_System_using_YOLOv5-and-CycleGAN\Streamlit_App\SOURCE\vgg_finetuned_model\kerasVggSigFeatures.h5", compile=False)
    
    
    
    
    
    feature_set = []
    # anchor image is resized to 256x256 to match outputs from gan.
    make_square(anchor_image)
    anchor_image_feature = extract_features(feature_extractor, anchor_image)
    test_images = [gan_op + image for image in os.listdir(gan_op) if image[2:6]=='fake']
    for image in test_images:
        test_image_feature = extract_features(feature_extractor, image)
        cosine_similarity = cosine_similarity_fn(anchor_image_feature, test_image_feature)
        cosine_similarity = round(cosine_similarity, 2)
        feature_set.append([image, cosine_similarity])
    return feature_set

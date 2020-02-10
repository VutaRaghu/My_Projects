# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 10:59:46 2019

@author: raghu
"""
pass
import warnings
warnings.filterwarnings("ignore")


json_file = open(r'C:\Users\raghu\Desktop\Projects\Face_Recoginition\faces\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

from tensorflow.keras.models import model_from_json
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")


import numpy as np
from tensorflow.keras.preprocessing import image
test_image=image.load_img(r'C:\Users\raghu\Desktop\Projects\Face_Recoginition\faces\srikanth\222.jpg',target_size=(200,200))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
#print(test_image)
result=loaded_model.predict(test_image)
classes = ['raghu','rajesh','srikanth']
a=list(result[0]);
b=a.index(max(a));
print(classes[b]);
#print(result)

from flask import *
app=Flask(__name__)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from werkzeug import secure_filename
from tensorflow.keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing import image
import warnings
import cv2
import numpy as np
import time
app.secret_key = "dnt tell" # for flash or alert
UPLOAD_FOLDER = 'upload/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def main():
    cap = cv2.VideoCapture(-1)
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    x=1
    while ret:
        ret, frame = cap.read()
        m =r'C:\Users\raghu\Desktop\Projects\Face_Recoginition\faces\upload\img' +  str(int(x)) + ".jpg"
        cv2.imwrite(m, frame)
        x=x+1
        print("Image Captured Sucessfully")
        break
    cv2.destroyAllWindows()
    cap.release()



def func():
    # -*- coding: utf-8 -*-
    """
    Created on Sat Dec 14 10:59:46 2019

    @author: raghu
    """
    
    warnings.filterwarnings("ignore")


    json_file = open(r'C:\Users\raghu\Desktop\Projects\Face_Recoginition\faces\model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    
    print("hi")
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model1.h5")
    print("Loaded model from disk")


    
    test_image=image.load_img(r'C:\Users\raghu\Desktop\Projects\Face_Recoginition\faces\upload\img1.jpg',target_size=(200,200))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    #print(test_image)
    result=loaded_model.predict(test_image)
    classes = ['raghu','rajesh','srikanth']
    a=list(result[0]);
    b=a.index(max(a));
    print(classes[b]);
    #print(result)
    
    return render_template("result.html",data=classes[b])
    

        
    '''
    loaded_model.predict(test_image,verbose=1)
    '''









@app.route('/')
def main_tem():
    return render_template("home.html")


@app.route('/Face_Recognition')
def Face_Recognition():
    return render_template("Face_Recognition.html")




@app.route('/Face_Recognition_1')
def Face_Recognition_1():
    return render_template("Face_Recognition_1.html")




@app.route('/Face_Recognition_insert',methods=['POST'])
def Face_Recognition_insert():
    if request.method == 'POST':
        b=main()
        p=func()
        return p



@app.route('/Face_Recognition_1_insert',methods=['POST'])
def Face_Recognition_1_insert():
	if request.method == 'POST':
		name= request.files['inputfile']
		filename=secure_filename(name.filename)
		d=name.filename.split('.')
		if d[-1]=='jpg' or 'jpeg':
			name.filename = "img1."+d[-1]
			name.save(os.path.join(app.config['UPLOAD_FOLDER'], name.filename))
			p=func()
			return p
		else:
			flash('Invalid File format')
			return render_template("Face_Recognition.html")


@app.route('/result')
def result():
    return render_template("result.html")



if __name__ =="__main__":
    app.run(debug=True)

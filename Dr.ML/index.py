def func(Hospt,Age,Time,AcuteT):
    #_____________________________________________________________________DATA_ACCESS_____________________________________________________________________________________
    path=r"D:\DrML_finalised\New folder\Veda_project\depressiondataset\depressiond1.csv"
    #path="Add Data path here.."
    names=["Hospt","Age","Time","AcuteT","Outcome"]
    data=pd.read_csv(path,names=names)
    #print(data.info())

    #___________________________________________________________________DATA_PREPROCESSING___________________________________________________________________________________
    #LABEL ENCODING............
    
    le=preprocessing.LabelEncoder()
    #data["Treat"]=le.fit_transform(data["Treat"])
    data["Outcome"]=le.fit_transform(data["Outcome"])
    #print(data.head())
    x=data.iloc[:,:4].values
    y=data.iloc[:,4].values
    """
    #____________________________________________________________________CORRELATONS______________________________________________________________________________________

    correlations=data.corr()
    print(correlations)
    #________________________________________________________________CORRELATION_PLOTTING________________________________________________________________________________
    fig=plt.figure()
    ax=fig.add_subplot(111)
    cax=ax.matshow(correlations,vmax=1,vmin=-1)
    fig.colorbar(cax)
    ticks=np.arange(0,6,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()

    #__________________________________________________________________HISTOGRAM_PLOTTING________________________________________________________________________________
    data.hist()
    plt.show()
    #________________________________________________________________SCATTER_MATRIX_PLOTTING_______________________________________________________________________________
    from pandas.plotting import scatter_matrix
    scatter_matrix(data)
    plt.show()
    """
    #______________________________________________________________________DATA_TRAINING____________________________________________________________________________________
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.312,random_state=1)
    """
    #_______________________________________________________________________NORMALIZATION_____________________________________________________________________________________
    from sklearn.preprocessing import Normalizer
    scaler=Normalizer().fit(x)
    rescaledx=scaler.transform(x)
    #print(rescaledx)

    #______________________________________________________________________STANDARDIZATION________________________________________________________________________________
    from sklearn import preprocessing
    names=data.columns
    scaler=preprocessing.StandardScaler()
    scaled_data=scaler.fit_transform(data)
    scaled_data=pd.DataFrame(scaled_data,columns=names)
    """
    #_______________________________________________________________________DECISION_TREE________________________________________________________________________________

    
    model_DT=tree.DecisionTreeClassifier(criterion="entropy", max_depth=1)
    model_DT=model_DT.fit(x_train,y_train)
    y_pred=model_DT.predict(x_test)
    #print(y_pred)
    y_pred1=model_DT.predict([[Hospt,Age,Time,AcuteT]])
    #print(y_pred1)
    if y_pred1== 0:
        data="You are Safe"
        flash('Details are processing')
        return render_template("Submit_Positive.html",data=data)
    else:
        data="You are effected by Depression"
        flash('Details are processing')
        return render_template("Submit_Negative.html",data=data)


    
    print("acciuracy DEC_TREE:",accuracy_score(y_test,y_pred)*100)

def func1(Pregnancies,GlucosePlasma,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    
    path =r"D:\DrML_finalised\New folder\Veda_project\KNN_diabetes\diabetes.csv"
    names=["Pregnancies","GlucosePlasma","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]

    data=pd.read_csv(path , names=names)
    #print(data.info(),("Data Information"))

    #r=pd.get_dummies(data["GlucosePlasma"],data["BloodPressure"],data["SkinThickness"])
    #print(r.head,("after"))

    X=data.iloc[:, :8] .values
    y=data.iloc[ :,8].values

    """
    sns.regplot(x="Pregnancies",y="Outcome",data=data)
    plt.show()
    correlations=data.corr()
    print(correlations)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    cax=ax.matshow(correlations,vmax=1,vmin=-1)
    fig.colorbar(cax)
    ticks=np.arange(0,6,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()
    """

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2897,random_state=4)
    '''
    #_________________________________________________NORMALIZATION_____________________________________________________________________________________
    from sklearn.preprocessing import Normalizer
    scaler=Normalizer().fit(X)
    rescaledx=scaler.transform(X)
    #print(rescaledx)

    #_________________________________________________STANDARDIZATION________________________________________________________________________________
    from sklearn import preprocessing
    names=data.columns
    scaler=preprocessing.StandardScaler()
    scaled_data=scaler.fit_transform(data)
    scaled_data=pd.DataFrame(scaled_data,columns=names)
    '''
    from sklearn.neighbors import KNeighborsClassifier
    model=KNeighborsClassifier(n_neighbors=17)
                               
    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)
    y_pred1=model.predict([[Pregnancies,GlucosePlasma,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])#this is new data given to the ml model
    #print(y_pred1,"= predict_data")
    if y_pred1==1:

        flash('Details are processing')
        data="You are effected by Diabetets"
        return render_template("Submit_Negative.html",data=data)
    else:

        data="You are Safe"
        flash('Details are processing')
        return render_template("Submit_Positive.html",data=data)



    from sklearn.metrics import accuracy_score
    print("accuracy:",accuracy_score(y_test,y_pred)*100)

def func2(male,age,current_smoker,cigsperday,bpmeds,prevalentscor,prevalentHyp,diabetets,totchol,sysbp,diaBP,bmi,heartrate,glucose):
    

    #__________________________________________________________________IMPORT_MODULES_____________________________________________________________________________________
    
    #_____________________________________________________________________DATA_ACCESS_____________________________________________________________________________________
    path=r"D:\DrML_finalised\New folder\Veda_project\Heart_disease_ML\heart_attack_prediction.csv"
    #path="Add Data path here.."
    names=["male","age","current_smoker","cigsperday","bpmeds","prevalentscor","prevalentHyp","diabetets","totchol","sysbp","diaBP","bmi","heartrate","glucose","tenyearchd"]
    data=pd.read_csv(path,names=names)
    #print(data.info())
    x=data.iloc[:,:14].values
    y=data.iloc[:,14].values
    """
    #___________________________________________________________________DATA_PREPROCESSING___________________________________________________________________________________
    Data processing is done manualy because some of the values in the data set are not avaliable .so they are replaced with some average values,
    so that our accuracy may not fall much!  
    """
    """
    
    #____________________________________________________________________CORRELATONS______________________________________________________________________________________

    correlations=data.corr()
    print(correlations)
    #________________________________________________________________CORRELATION_PLOTTING________________________________________________________________________________
    fig=plt.figure()
    ax=fig.add_subplot(111)
    cax=ax.matshow(correlations,vmax=1,vmin=-1)
    fig.colorbar(cax)
    ticks=np.arange(0,9,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()

    #__________________________________________________________________HISTOGRAM_PLOTTING________________________________________________________________________________
    data.hist()
    plt.show()
    #________________________________________________________________SCATTER_MATRIX_PLOTTING_______________________________________________________________________________
    from pandas.plotting import scatter_matrix
    scatter_matrix(data)
    plt.show()
    """
    #______________________________________________________________________DATA_TRAINING____________________________________________________________________________________
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.312,random_state=1)

    #We have done normalization and standardization but haven't got any differences in accurarcies .So we have removed these..
    #_______________________________________________________________________NORMALIZATION_____________________________________________________________________________________
    from sklearn.preprocessing import Normalizer
    scaler=Normalizer().fit(x)
    rescaledx=scaler.transform(x)
    #print(rescaledx)
    """
    #______________________________________________________________________STANDARDIZATION________________________________________________________________________________
    from sklearn import preprocessing
    names=data.columns
    scaler=preprocessing.StandardScaler()
    scaled_data=scaler.fit_transform(data)
    scaled_data=pd.DataFrame(scaled_data,columns=names)
    """
    """
    #_______________________________________________________________________KNN____________________________________________________________________________________________
    #Got an accuracy of  84.05139833
    from sklearn.neighbors import KNeighborsClassifier
    model_knn=KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(x_train,y_train) 
    y_pred=model_knn.predict(x_test)
    from sklearn.metrics import accuracy_score
    print("acciuracy KNN:",accuracy_score(y_test,y_pred)*100)

    #_______________________________________________________________________NAIVE_BAYES____________________________________________________________________________________
    #Got an accuracy of 82.9176115
    from sklearn.naive_bayes import GaussianNB
    model_nb=GaussianNB()
    model_nb.fit(x_train,y_train)
    y_pred=model_nb.predict(x_test)
    print("accuracy NAIVE:",accuracy_score(y_test,y_pred)*100)
    #_______________________________________________________________________DECISION_TREE________________________________________________________________________________
    #Got an accuracy of 75.58578987150416
    from sklearn import tree
    model_DT=tree.DecisionTreeClassifier(criterion="entropy")
    model_DT=model_DT.fit(x_train,y_train)
    y_pred=model_DT.predict(x_test)
    #print(y_pred)
    print("acciuracy DEC_TREE:",accuracy_score(y_test,y_pred)*100)


    #_______________________________________________________________________SVM_________________________________________________________________________________________________________
    #Gave an accuracy of 86.9992441421019
    from sklearn.svm import SVC
    model_svm=SVC()
    model_svm.fit(x_train,y_train)
    y_pred_svm=model_svm.predict(x_test)
    print("svm accuracy:",accuracy_score(y_test,y_pred_svm)*100)

    #______________________________________________________________________LOGISTIC_REGRESSION_____________________________________________________________________________
    #This algorithm gave the maximum accuracy!!
    #The maximum accuracy we got is ...    87.07482993197279
    #So  by using LOGISTIC_REGRESSION we got better results...
    from sklearn.linear_model import LogisticRegression
    model_LOGI=LogisticRegression()
    model_LOGI.fit(x_train,y_train)
    y_pred=model_LOGI.predict(x_test)
    from sklearn.metrics import accuracy_score
    print("acciuracy LOGISTIC:",accuracy_score(y_test,y_pred)*100)

    #New test data
    y_pred=model_LOGI.predict([[0,61,1,30,1,0,1,1,225,150,95,28.58,65,103]])
    print(y_pred)
    #we got future prediction is 0(no chances of heart disease) with LogisticRegression. 

    """
    #________________________________________________________________RANDOM_FOREST__________________________________________________________________________________________
    #Among all these three random forest techniques the highest accuracy of 86.77248677248677

    from sklearn.ensemble import RandomForestClassifier
    """
    model5=RandomForestClassifier(n_estimators=200)                          
    model5.fit(x_train,y_train)
    y_pred_random_forest=model5.predict(x_test)
    from sklearn.metrics import accuracy_score
    print("Random_forest accuracy:",accuracy_score(y_test,y_pred_random_forest)*100)
    """
    modelrand=RandomForestClassifier(max_depth=5,min_samples_split=3,min_samples_leaf=1)
    modelrand.fit(x_train,y_train)
    y_pred_random_forest=modelrand.predict(x_test)
    from sklearn.metrics import accuracy_score
    print("Random_forest accuracy_leaf:",accuracy_score(y_test,y_pred_random_forest)*100)
    """
    modelrand1=RandomForestClassifier(max_features=10)
    modelrand1=RandomForestClassifier(max_features=.3)
    modelrand1.fit(x_train,y_train)
    y_pred_random_forest=modelrand1.predict(x_test)
    print("Random_forest accuracy_features:",accuracy_score(y_test,y_pred_random_forest)*100)
    """

    y_pred1=modelrand.predict([[male,age,current_smoker,cigsperday,bpmeds,prevalentscor,prevalentHyp,diabetets,totchol,sysbp,diaBP,bmi,heartrate,glucose]])
    print(y_pred1)
    if y_pred1==1:
        data="You are effected by Heart disease"
        flash('Details are processing')
        return render_template("Submit_Negative.html",data=data)
    else:
 
        data="You are Safe"
        flash('Details are processing')
        return render_template("Submit_Positive.html",data=data)







def func3():
    # -*- coding: utf-8 -*-
    """
    Created on Tue Sep  3 15:19:29 2019
    """

        # -*- coding: utf-8 -*-
    """
    Created on Tue Sep  3 15:19:29 2019

    @author: Sathish
    """
    warnings.filterwarnings("ignore")

    # load json and create model
    json_file = open(r'D:\DrML_finalised\New folder\Dr_ML\model1.json', 'r')
    loaded_model1_json = json_file.read()
    json_file.close()

    from keras.models import model_from_json
    loaded_model = model_from_json(loaded_model1_json)

    # load weights into new model
    loaded_model.load_weights("model1.h5")
    print("Loaded model from disk")



    

    
    #k = int(input("Select one \n 1)Predict normal Way \n press any key to Use dynmic way \n ..")
    #test_image=image.load_img(r"E:\Desktop\New folder\Dr_ML\upload\img1.jpg",target_size=(75,100))
    
    test_image=image.load_img(r"D:\DrML_finalised\New folder\Dr_ML\upload\img1.jpg",target_size=(75,100))
    test_image0=image.img_to_array(test_image)
    test_image1=np.expand_dims(test_image0,axis=0)
    result=loaded_model.predict(test_image1)

    '''
    validation_generator.class_indices
    '''
    print(result)

    


    if result[0][0]==1:
        data="You are effected by Skin_Cancer"
        flash('Details are processing')
        return render_template("Submit_Negative.html",data=data)
    else:
 
        data="You are not effected"
        flash('Details are processing')
        return render_template("Submit_Positive.html",data=data)

        
    '''
    loaded_model.predict(test_image,verbose=1)
    '''

def func4():
    warnings.filterwarnings("ignore")

    # load json and create model
    json_file = open(r'D:\DrML_finalised\New folder\Dr_ML\model1.json', 'r')
    loaded_model1_json = json_file.read()
    json_file.close()

    
    loaded_model = model_from_json(loaded_model1_json)

    # load weights into new model
    loaded_model.load_weights("model1.h5")
    print("Loaded model from disk")



    

    

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = r"E:\Desktop\New folder\Dr_ML\upload\img1.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))

    cam.release()
    del(cam)
    cv2.destroyAllWindows()
    test_image=image.load_img(r"E:\Desktop\New folder\Dr_ML\upload\img1.jpg",target_size=(75,100))
    test_image0=image.img_to_array(test_image)
    test_image1=np.expand_dims(test_image0,axis=0)
    result=loaded_model.predict(test_image1)

    '''
    validation_generator.class_indices
    '''
    print(result)

    


    if result[0][0]==1:
        data="You are effected by Skin_Cancer"
        flash('Details are processing')
        return render_template("Submit_Negative.html",data=data)
    else:
 
        data="You are not effected"
        flash('Details are processing')
        return render_template("Submit_Positive.html",data=data)






from flask import *
app=Flask(__name__)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from werkzeug import secure_filename
import warnings
from keras.models import model_from_json
from keras.preprocessing import image
import cv2
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

app.secret_key = "dnt tell" # for flash or alert
UPLOAD_FOLDER = 'upload/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def main_tem():
    return render_template("Dr_ML.html")

@app.route('/home')
def home():
    return render_template("home.html")


@app.route('/depression')
def index():
    return render_template("depression.html")


@app.route('/depression_insert',methods=['GET','POST'])
def depression_insert():
    Hospt = request.form['Hospt']
    Age = request.form['Age']
    Time = request.form['Time']
    AcuteT = request.form['AcuteT']
    
    p=func(Hospt,Age,Time,AcuteT)
    return p
    

@app.route("/diabeties")
def diabeties():
    return render_template("diabeties.html")   

@app.route('/diabeties_insert',methods=['GET','POST'])
def diabeties_insert():
    Pregnancies = request.form['Pregnancies']
    GlucosePlasma = request.form['GlucosePlasma']
    BloodPressure = request.form['BloodPressure']
    SkinThickness = request.form['SkinThickness']
    Insulin = request.form['Insulin']
    BMI = request.form['BMI']
    DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
    Age = request.form['Age']
    
    p=func1(Pregnancies,GlucosePlasma,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    return p
     

@app.route('/heart')
def heart():
    return render_template("heart.html")


@app.route('/heart_insert',methods=['GET','POST'])




def heart_insert():
    male = request.form['male']
    age = request.form['age']
    current_smoker = request.form['current_smoker']
    cigsperday = request.form['cigsperday']
    bpmeds = request.form['bpmeds']
    prevalentscor = request.form['prevalentscor']
    prevalentHyp = request.form['prevalentHyp']
    diabetets = request.form['diabetets']
    totchol = request.form['totchol']
    sysbp = request.form['sysbp']
    diaBP = request.form['diaBP']
    bmi = request.form['bmi']
    heartrate = request.form['heartrate']
    glucose = request.form['glucose']
    
    p=func2(male,age,current_smoker,cigsperday,bpmeds,prevalentscor,prevalentHyp,diabetets,totchol,sysbp,diaBP,bmi,heartrate,glucose)
    return p
    
     
    






@app.route('/Skin_Cancer')
def Skin_Cancer():
    return render_template("Skin_Cancer.html")


@app.route('/Skin_Cancer_insert',methods=['GET','POST'])
def Skin_Cancer_insert():
	p=func4()
	return p


@app.route('/Skin_Cancer_1')
def Skin_Cancer_1():
    return render_template("Skin_Cancer_1.html")


@app.route('/Skin_Cancer_1_insert',methods=['GET','POST'])
def Skin_Cancer_1_insert():
	if request.method == 'POST':
		name= request.files['inputfile']
		filename=secure_filename(name.filename)
		d=name.filename.split('.')
		if d[-1]=='jpg':
			name.filename = "img1."+d[-1]
			name.save(os.path.join(app.config['UPLOAD_FOLDER'], name.filename))
			p=func3()
			return p
		else:
			flash('Invalid File format')
			return render_template("skin-cancer.html")



    


if __name__ =="__main__":
    app.run(debug=True)

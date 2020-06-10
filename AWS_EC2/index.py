def func(Hospt,Age,Time,AcuteT):
    loaded_model = pickle.load(open('depression.sav', 'rb'))
    y_pred1 = loaded_model.predict([[Hospt,Age, Time,AcuteT]])
    if y_pred1== 0:
        data="You are safe according to the given data"
        
    else:
        data="You might be effected by depression in the future based on data processed..."

    return render_template("depression.html",data=data)



def func1(Pregnancies,GlucosePlasma,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    loaded_model = pickle.load(open('diabeties.sav', 'rb'))
    y_pred1 = loaded_model.predict([[Pregnancies,GlucosePlasma,BloodPressure , SkinThickness, Insulin,  BMI,  DiabetesPedigreeFunction,Age]])
    
    if y_pred1==1:
        data="You might be effected by Diabeties in the future based on data processed..."
        
    else:
        data="You are safe according to the given data"

    return render_template("diabeties.html",data=data)





def func2(male,age,current_smoker,cigsperday,bpmeds,prevalentscor,prevalentHyp,diabetets,totchol,sysbp,diaBP,bmi,heartrate,glucose):
    
    loaded_model = pickle.load(open('heart.sav', 'rb'))
    y_pred1 = loaded_model.predict([[male, age, current_smoker,   cigsperday,   bpmeds,   prevalentscor,  prevalentHyp,diabetets,totchol,sysbp,diaBP,bmi,heartrate,glucose]])
    if y_pred1==1:
        data="You might be effected by Heart Diseases in the future based on data processed..."

    else:
 
        data="You are safe according to the given data"
    return render_template("heart.html",data=data)




from flask import *
app=Flask(__name__)
import pickle

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
    
     



if __name__ =="__main__":
    app.run(host='0.0.0.0',port=8080)
    #app.run(debug=True)

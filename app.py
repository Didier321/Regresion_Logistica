from flask import Flask, render_template, request
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), 'models', 'modelo_regresion.pkl')
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    SepalLengthCm	= float(request.form['SepalLengthCm'])
    SepalWidthCm	= float(request.form['SepalWidthCm'])
    PetalLengthCm	= float(request.form['PetalLengthCm'])
    PetalWidthCm	= float(request.form['PetalWidthCm'])
    
    pred_probabilities = model.predict_proba([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
    
    class_names = model.classes_
    
    mensaje = ""
    for i, class_name in enumerate(class_names):
        prob = pred_probabilities[0, i] * 100
        mensaje += f"Probabilidad de {class_name}: {round(prob, 2)}% <br/>"
        
    return render_template('result.html', pred=mensaje)

if __name__ == '__main__':
    app.run()
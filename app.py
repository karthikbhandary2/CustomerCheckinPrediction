import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model1.pkl','rb'))#loading the model
trans1 = pickle.load(open('transform1.pkl', 'rb'))#Loding the encoder
trans2 = pickle.load(open('transform2.pkl', 'rb'))#Loading the encoder
scale = pickle.load(open('scale.pkl', 'rb'))#Loading the scaler
@app.route('/')
def home():
    return render_template('index.html')#rendering the home page

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    print(features)
    features[11] = trans1.transform([features[11]])
    features[12] = trans2.transform([features[12]])
    features_to_scale = [features[0], features[1], features[2], features[3], features[4], features[9], features[10]]
    features_to_scale = np.array(features_to_scale).reshape(1,-1)
    # return render_template('index.html', prediction_text="shape: {}".format(features_to_scale.shape))
    features_to_scale = scale.transform(features_to_scale)
    # return render_template('index.html', prediction_text="shape: {}".format(features_to_scale))
    for i in [0,1,2,3,4,9,10]:
        for j in features_to_scale:
            for z in j: 
                features[i] = z
    final_features = [np.array(features, dtype=float)]
    # final_features = final_features[None, ...]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    # output = len(prediction)

    return render_template('index.html', prediction_text='Booked: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
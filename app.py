from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
import tensorflow as tf
import pickle, re
import numpy as np
import sklearn
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Flask and Swagger endpoint
app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info = {
        'title' : LazyString(lambda: 'API Documentation for LSTM and Neural Network model'),
        'version' : LazyString(lambda: '1.0.0'),
        'description' : LazyString(lambda: 'Dokumentasi API untuk model LSTM and Neural Network')
    },
    host = LazyString(lambda: request.host)
)

swagger_config = {
    'headers': [],
    'specs': [
        {
            'endpoint': 'docs',
            'route': '/docs.json',
        }
    ],
    'static_url_path': "/flasgger_static",
    'swagger_ui': True,
    'specs_route': "/docs/"
}
swagger = Swagger(app, template=swagger_template,             
                  config=swagger_config)

#Tools to run the function 
max_features = 100000
sentiment = ['negative', 'neutral', 'positive']
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
#===================================================================
#Load Feature Extraction for LSTM
file = open("resource_LSTM/x_pad_sequences.pickle",'rb')
feature_file_from_lstm = pickle.load(file)
file.close()

#Load Model for LSTM
model_LSTM = load_model('model_LSTM/model_pas.h5')
#===================================================================
#Vectorizer For Neural Network
count_vect = pickle.load(open("resource_NN/feature.p","rb"))
#===================================================================
#Load Model for Neural Network
model_NN = pickle.load(open("modenl_NN/model.p","rb"))
#===================================================================



#Cleansing Function
def cleansing(sent):

    string = sent.lower()

    string = re.sub(r'[^a-zA-Z0-9]',' ',string)
    string = re.sub('rt',' ',string) # Remove every retweet symbol
    string = re.sub('user',' ',string) # Remove every username
    string = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',string) # Remove every URL
    return string



#===================================================================

#Func API LSTM(Text)
@swag_from("docs/lstm_text.yml",methods=['POST'])
@app.route('/lstm_text',methods=['POST'])
def lstm_text():
    #Request text 
    original_text = request.form.get('text')
    #Cleansing
    text = [cleansing(original_text)]
    
    #Feature Extraction
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    
    #Prediction
    prediction = model_LSTM.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    json_response = {
      'status_code' : 200,
        'description' : "Result of Sentiment Analysis using LSTM",
        'data' : {
            'text' : original_text,
            'sentiment' : get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

#===================================================================

#Func API LSTM(Text)
@swag_from("docs/lstm_file.yml",methods=['POST'])
@app.route('/lstm_file',methods=['POST'])
def lstm_file():
    # Upladed file
    file = request.files['file']

    # Import file csv ke Pandas
    df = pd.read_csv(file,header=0)
    #Cleansing
    df['text_clean'] = df.apply(lambda row : cleansing(row['text']), axis = 1)
    
    result = []

    # Feature Extraction & Predict Data
    for index, row in df.iterrows():
        text = tokenizer.texts_to_sequences([(row['text_clean'])])
        feature = pad_sequences(text, maxlen=feature_file_from_lstm.shape[1])
        prediction = model_LSTM.predict(feature)
        #predict the sentiment 
        get_sentiment = sentiment[np.argmax(prediction[0])]
        # append sentiment to result
        result.append(get_sentiment)

    # Get result from file in "List" format
    original = df.text_clean.to_list()

    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis from csv file using LSTM",
        'data' : {
            'text' : original,
            'sentiment' : result
        },
    }

    response_data = jsonify(json_response)
    return response_data

#===================================================================
#Func API Neural Network (text)
@swag_from("docs/nn_text.yml", methods=['POST'])
@app.route('/nn_text', methods=['POST'])
def nn_text():
    #Request text 
    original_text = request.form.get('text')
    #Cleansing text
    clean_text = [cleansing(original_text)]
    #Vectorizing 
    text = count_vect.transform(clean_text)
    #Predict sentiment
    result = model_NN.predict(text)[0]

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using Neural Network",
        'data' : {
            'text' : original_text,
            'sentiment' : result 
        },
    }

    response_data = jsonify(json_response)
    return response_data

#===================================================================
#Func API for Neural Netwrok(File)
@swag_from("docs/nn_file.yml", methods=['POST'])
@app.route('/nn_file', methods=['POST'])
def nn_file():
    #upload file
    file = request.files['file']
    #Import file to pandas DataFrame
    df = pd.read_csv(file,header=0)
    #Cleansing text
    df['text_clean'] = df.apply(lambda row : cleansing(row['text']), axis = 1)

    result = []
    #Vectorizing & Predict sentiment
    for index, row in df.iterrows():
        text = count_vect.transform([(row['text_clean'])])

        #append predicted sentiment to result 
        result.append(model_NN.predict(text)[0])
        
    # Get result from file in "List" format
    original_text = df.text_clean.to_list()

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using Neural Network",
        'data': {
            'text' : original_text,
            'sentiment' : result
        },
    }

    response_data = jsonify(json_response)
    return response_data

#===================================================================
#Run API 
if __name__ == '__main__' :
    app.run(debug=True)
import flask
import joblib
import pickle
import json
import pandas as pd
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import regex as re
with open(f'model/model.json', 'r') as f:
    model= f.read()
    model = model_from_json(model)
    model.load_weights('model/classifier.h5')
#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
x=[]
positive_tweets=0
negative_tweets=0
tokenizer=Tokenizer(num_words=40000)
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
app = flask.Flask(__name__, template_folder='templates')
@app.route('/',methods=['GET','POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method=='POST':
        #tweet=flask.request.args.get('tweet','')
        tweet=flask.request.form.get('tweet')
        tweet=np.array([tweet])
        for words in tweet:
            x.append(preprocess_text(words))
        tokenizer.fit_on_texts(x)
        input_variables=tokenizer.texts_to_sequences(x)
        input_variables=pad_sequences(input_variables,padding='post',maxlen=50)
        #input_variables=flask.request.args.get('input_variables','')
        prediction=(model.predict(input_variables)>0.5).astype(int)
        print(prediction)
        if prediction[-1]==1:
           prediction="Thanks for your negative feedback.We will try to improve"
           #negative_tweets+=1
        else:
           prediction="Thanks for your positive feedback"
           #positive_tweets+=1
        return flask.render_template('main.html',original_input={'Tweet':tweet},result=prediction)
        #print("no of positive tweets:",positive_tweets)
        #print("no of negative tweets:",negative_tweets)
        #return redirect(url_for("127.0.0.1:5000"))
if __name__ == '__main__':
    app.run(threaded=False)
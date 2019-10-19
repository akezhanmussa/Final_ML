from flask import Flask, render_template, Response, request, redirect, flash, url_for
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import os
import cv2
import sys
sys.path.append('/Users/akemussa/Desktop/ml/ML400/final_project/')
import Final_ML.test as test
import numpy as np
from keras.utils import to_categorical


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, template_folder =  './bin')
Bootstrap(app)

def analyzeFrame(frame):
    return test.give_label(frame)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods = ['GET', 'POST'])
def homePage():
    filename = ""
    uploaded = 0
    final_label = ""
    #Upload file
    if request.method == 'POST':
        if 'Image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['Image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join("./data", filename)
            file.save(filepath)
            frame = cv2.imread(filepath)
            frame = cv2.resize(frame, (224,224))
            frame = frame/255
            X = []
            X.append(frame)
            X = np.array(X)
            label = analyzeFrame(X)
            print("PREDICTED LABEL IS ", label)
            label1 = round(label[0][0], 3)
            label2 = round(label[0][1], 3)
            final_label = "Prediction that it is not cactus is {}, and {} otherwise".format(label1, label2)
            uploaded = 1

    return render_template("index.html", filename = filename, uploaded = uploaded, label = final_label)


if __name__ == '__main__':
    app.run(host = "localhost", port = 8080)

#Usage: python app.py
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import joblib
import time
import uuid
import base64
import matplotlib.pyplot as plt
from datetime import timedelta

#img_width, img_height = 150, 150
model_path = './models/one_all.model'

#model = joblib.load(model_path)
#model.load_weights(model_weights_path)

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['csv'])
TYPE = ['A1','A2','A2-','A3','A3+']

def load_model():
    global model
    model = joblib.load(model_path)

def get_as_base64(url):
    return base64.b64encode(request.get(url).content)

def predict(file):
    # Add
    file = pd.read_csv(file,encoding='gbk')
    # mean = -1.6601907659274462e-24
    # std = 1.1359138229621789e-21
    # x -= mean
    # x /= std
    # x = x.clip(-2,2)
    x = file.iloc[:,3:20]
    array = model.predict_proba(x)
    print(array)
    print(type(array))
    print(array.shape)
    return array

def read_info(file):
    file = pd.read_csv(file,encoding='gbk')
    info = file.iloc[:,:3]
    return info

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def plot_value_array(predictions_array):
    plt.grid(False)
    plt.xticks(range(5),TYPE)
    plt.yticks([])
    thisplot = plt.bar(range(5), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('blue')

def plot_image(file):
    pic_name = my_random_string(6)+file[:-4]+'.png'
    data = pd.read_csv(UPLOAD_FOLDER+file).to_numpy()
    data_small = data
    plot_value_array(data_small)
    figure = {'family':'Times New Roman','weight':'normal','size':17}
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], pic_name),bbox_inches = 'tight')
    return pic_name


app = Flask(__name__, static_url_path="")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#@app.route("/")
#def template_test():
#    return render_template('template.html', label='', imagesource='./uploads/template.png')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['uploadfile']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # pic_file = plot_image(filename)
            result = predict(file_path)
            info = read_info(file_path)
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            
            
            Jijiu_num = info.iloc[0,0]
            Time = info.iloc[0,1]
            Category = info.iloc[0,2]

            result = np.around(result,decimals=4)*100
            print(result)
            result=result[0]
            # result_sort = np.argsort(-result)
            label1 = str(TYPE[0])+' : '+str(result[0])+"%"
            label2 = str(TYPE[1])+' : '+str(result[1])+"%"
            label3 = str(TYPE[2])+' : '+str(result[2])+"%"
            label4 = str(TYPE[3])+' : '+str(result[3])+"%"
            label5 = str(TYPE[4])+' : '+str(result[4])+"%"
            return render_template('template.html',Jijiu_num=Jijiu_num,Time=Time,Category=Category, label1=label1,label2=label2,label3=label3,label4=label4,label5=label5)
    elif request.method == "GET":
        return render_template('template.html', label='')

from flask import send_from_directory
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug.middleware.shared_data import SharedDataMiddleware

app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})



if __name__ == "__main__":
    app.debug=True
    load_model()
    # set main path of static folder
    app._static_folder ='/home/wangzj/data/web-api-basewine/test/'
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
    app.run(host='0.0.0.0', port=3003)
    # app.run()

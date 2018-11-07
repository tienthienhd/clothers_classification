import os
from flask import Flask, render_template, request, jsonify, url_for
from models import Models
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
models = Models()


def jsonify_str(output_list):
    with app.app_context():
        with app.test_request_context():
            result = jsonify(output_list)

    return result


@app.route('/')
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            # return redirect(request.url)
            return jsonify_str({'error': 'Cannot get image'})
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return jsonify_str({'error': 'Cannot get image'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            url_local = os.path.join('static/image_test', filename)
            file.save(url_local)
            models.set_image_url(url_local)
            return 'Ok'


@app.route("/download_file", methods=['GET', 'POST'])
def download_file():
    if request.method == "POST":
        try:
            image_url = request.form['url']
            print('received url:', image_url)
            url_local = models.download(image_url)
            models.set_image_url(url_local)
            return url_local
        except Exception as e:
            return jsonify_str(
                {'error': 'Cannot downloaded image with url:' + image_url})


@app.route("/query", methods=['GET', 'POST'])
def query():

        if request.method == "POST":
            if models.current_image_url is None:
                return jsonify_str({'error': 'Please upload or add link image.'})
            result = models.predict()

            response = {
                'svm': result[0],
                'knn': result[1],
                'cnn': result[2]
            }
            print(response)
            return jsonify_str(response)

app.run(host='0.0.0.0', port=5000, debug='development')
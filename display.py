
from flask import Flask, render_template, request, jsonify
from models import Models

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


@app.route("/query", methods=['GET', 'POST'])
def query():

        if request.method == "POST":
            image_url = request.form['url']
            print('received url:', image_url)
            # return image_url
            try:
                result = models.predict(image_url)

                response = {
                    'svm': result[0],
                    'knn': result[1],
                    'cnn': result[2]
                }
                print(response)
                return jsonify_str(response)
            except Exception as e:
                return jsonify_str({'error': 'Cannot downloaded image with url:' + image_url})

app.run(host='0.0.0.0', port=5000, debug='development')
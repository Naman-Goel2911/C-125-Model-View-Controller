from flask import Flask, jsonify, request
from classifier import getPred

app = Flask(__name__)

@app.route('/pred-data', methods = ['POST'])

def pred_data():
    image = request.files.get('alphabet')
    prediction = getPred(image)
    return jsonify({
        'prediction':prediction
    }),200

if (__name__ == '__main__'):
    app.run(debug=True)
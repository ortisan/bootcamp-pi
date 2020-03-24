from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/api/recommendations', methods=['POST', 'GET'])
def recommendation_user_id():
    job = request.args.get('user_id')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    resp = [{'timestamp': timestamp}]
    return jsonify(resp)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

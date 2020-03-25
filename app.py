from flask import Flask, request, jsonify
from datetime import datetime
from recommendations import SimilarUsers, SimilarProducts, SimilarProductsUsers

app = Flask(__name__)

similarProductUsers = SimilarProductsUsers()

@app.route('/')
def health_check():
    return 'Running...'


@app.route('/api/recommendations', methods=['POST', 'GET'])
def recommendation_user_id():
    user_id = request.args.get('user_id')
    try:
        products = similarProductUsers.neighbors_product_idx(user_id)
        return jsonify(products)
    except Exception as exc:
        return ("Error: {0}".format(exc))
    # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # resp = [{'timestamp': timestamp}]
    


if __name__ == '__main__':    
    app.run(debug=True, host='0.0.0.0')

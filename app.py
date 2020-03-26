from flask import Flask, request, jsonify
from datetime import datetime
from recommendations import SimilarUsers, SimilarProducts, SimilarProductsUsers
from preprocess import PreProcessDataset1, PreProcessDataset3

app = Flask(__name__)

similarProductUsers = SimilarProductsUsers()
preProcessDataset1 = PreProcessDataset1()
preProcessDataset3 = PreProcessDataset3()


@app.route('/')
def health_check():
    return 'Running...'


@app.route('/api/recommendations', methods=['POST', 'GET'])
def recommendation_user_id():
    user_id = request.args.get('user_id')
    try:
        list_products_id = similarProductUsers.neighbors_products(user_id)
        user = preProcessDataset1.get_user_information_by_id(user_id)
        products = preProcessDataset3.get_products_information_by_id(list_products_id)
        return jsonify({"user": user, "products": products})
    except Exception as exc:
        return ("Error: {0}".format(exc))


if __name__ == '__main__':    
    app.run(debug=True, host='0.0.0.0')

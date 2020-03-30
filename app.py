from flask import Flask, request, jsonify
from recommendations import SimilarUsers, SimilarProducts, SimilarProductsUsers, QuantityProductRegression
from preprocess import PreProcessDataset1, PreProcessDataset3
from helpers import from_dataframe_to_list_dict

app = Flask(__name__)

similarProductUsers = SimilarProductsUsers()
preProcessDataset1 = PreProcessDataset1()
preProcessDataset3 = PreProcessDataset3()
quantityProductRegression = QuantityProductRegression()


@app.route('/')
def health_check():
    return 'Running...'


@app.route('/api/recommendations', methods=['POST', 'GET'])
def recommendation_user_id():
    user_id = request.args.get('user_id')
    try:
        dict_products_id_similarity = similarProductUsers.neighbors_products(user_id)
        user = preProcessDataset1.get_user_information_by_id([user_id])

        df_products = preProcessDataset3.get_products_information_by_id(dict_products_id_similarity)
        mean_estimated_risk = df_products.RiscoAtivo__c.mean()
        user[0].update({'mean_estimated_risk': mean_estimated_risk})
        df_regression = quantityProductRegression.fit_transform(user_id, df_products)

        list_products = from_dataframe_to_list_dict(df_regression)

        return jsonify({"products": list_products, "user": user})
    except Exception as exc:
        return "Error: {0}".format(exc)


if __name__ == '__main__':    
    app.run(debug=True, host='0.0.0.0')

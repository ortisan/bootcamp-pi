from flask import Flask, request, jsonify
from recommendations import SimilarUsers, SimilarProducts, SimilarProductsUsers, QuantityProductRegression
from preprocess import PreProcessDataset1, PreProcessDataset3
from helpers import from_dataframe_to_list_dict

app = Flask(__name__)

similarProductUsers = SimilarProductsUsers()
similarUsers = SimilarUsers()
preProcessDataset1 = PreProcessDataset1()
preProcessDataset3 = PreProcessDataset3()
quantityProductRegression = QuantityProductRegression()


@app.route('/')
def health_check():
    return 'Running...'


@app.route('/api/recommendations/user_to_product', methods=['POST', 'GET'])
def recommendation_user_to_product():
    user_id = request.args.get('user_id')
    try:
        list_current_product_id = preProcessDataset1.get_user_current_products([user_id])

        # user to product recommendations
        dict_products_id_similarity = similarProductUsers.neighbors_products(user_id, list_current_product_id)
        df_recommended_products = preProcessDataset3.get_products_information_by_id(dict_products_id_similarity.keys())
        df_recommended_products = preProcessDataset3.\
            add_similarity_column(df_recommended_products, dict_products_id_similarity)
        mean_recommended_risk = df_recommended_products.RiscoAtivo__c.mean()

        # current information of the user products
        df_current_products = preProcessDataset3.get_products_information_by_id(list_current_product_id)
        mean_current_risk = df_current_products.RiscoAtivo__c.mean()

        # current information about the user
        df_user = preProcessDataset1.get_user_information_by_id([user_id])
        user = from_dataframe_to_list_dict(df_user)
        user[0].update({'mean_estimated_risk': mean_recommended_risk, 'mean_current_risk': mean_current_risk})

        # how many products should the user buy
        df_regression = quantityProductRegression.fit_transform(user_id, df_recommended_products)

        list_products = from_dataframe_to_list_dict(df_regression)
        return jsonify({"recommended_products": list_products, "user": user})
    except Exception as exc:
        return "Error: {0}".format(exc)


@app.route('/api/recommendations/user_to_user', methods=['POST', 'GET'])
def recommendation_user_to_user():
    user_id = request.args.get('user_id')
    try:
        list_current_product_id = preProcessDataset1.get_user_current_products([user_id])

        # similar products by users
        dict_users_id_similarity = similarUsers.neighbors_user_id([user_id], n_closest=3)
        list_similar_users_product_id = preProcessDataset1. \
            get_user_current_products(list(dict_users_id_similarity.keys()))
        df_neigh_products = preProcessDataset3.get_products_information_by_id(list_similar_users_product_id)
        list_neigh_products = from_dataframe_to_list_dict(df_neigh_products)
        mean_neighbors_risk = df_neigh_products.RiscoAtivo__c.mean()

        # current information of the user products
        df_current_products = preProcessDataset3.get_products_information_by_id(list_current_product_id)
        mean_current_risk = df_current_products.RiscoAtivo__c.mean()

        # current information about the user
        df_user = preProcessDataset1.get_user_information_by_id([user_id])
        user = from_dataframe_to_list_dict(df_user)
        user[0].update({'mean_neighbors_risk': mean_neighbors_risk, 'mean_current_risk': mean_current_risk})

        # how many products should the user buy
        df_regression = quantityProductRegression.fit_transform(user_id, df_neigh_products)

        list_products = from_dataframe_to_list_dict(df_regression)
        return jsonify({"neighbor_products": list_products, "user": user})
    except Exception as exc:
        return "Error: {0}".format(exc)


if __name__ == '__main__':    
    app.run(debug=True, host='0.0.0.0')

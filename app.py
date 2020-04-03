from flask import Flask, request, jsonify
from recommendations import SimilarUsers, SimilarProducts, SimilarProductsUsers, QuantityProductRegression, SimilarityEmbeddings
from preprocess import PreProcessDataset1, PreProcessDataset3
from helpers import from_dataframe_to_list_dict

app = Flask(__name__)

# similarityEmbedding = None
# similarProductUsers = None
# similarUsers = None
# similarProducts = None
# preProcessDataset1 = None
# preProcessDataset3 = None
# quantityProductRegression = None

@app.route('/')
def health_check():
    return 'Running...'


# @app.route('/api/recommendations/user_to_product', methods=['POST', 'GET'])
# def recommendation_user_to_product():
#     user_id = request.args.get('user_id')
#     try:
#         list_current_product_id = preProcessDataset1.get_user_current_products([user_id])
#         print(list_current_product_id)
#         # user to product recommendations
#         dict_products_id_similarity = similarProductUsers.neighbors_products(user_id, list_current_product_id, n_closest=6)
#         print(dict_products_id_similarity)
#         df_recommended_products = preProcessDataset3.get_products_information_by_id(dict_products_id_similarity.keys())
#         print(df_recommended_products)
#         df_recommended_products = preProcessDataset3.\
#             add_similarity_column(df_recommended_products, dict_products_id_similarity)
#         mean_recommended_risk = df_recommended_products.RiscoAtivo__c.mean()

#         # current information of the user products
#         df_current_products = preProcessDataset3.get_products_information_by_id(list_current_product_id)
#         mean_current_risk = df_current_products.RiscoAtivo__c.mean()

#         # current information about the user
#         df_user = preProcessDataset1.get_user_information_by_id([user_id])
#         user = from_dataframe_to_list_dict(df_user)
#         user[0].update({'mean_estimated_risk': mean_recommended_risk, 'mean_current_risk': mean_current_risk})

#         # how many products should the user buy
#         df_regression = quantityProductRegression.fit_transform(user_id, df_recommended_products)

#         list_products = from_dataframe_to_list_dict(df_regression)
#         return jsonify({"recommended_products": list_products, "user": user})
#     except Exception as exc:
#         return "Error: {0}".format(exc)


# @app.route('/api/recommendations/user_to_user', methods=['POST', 'GET'])
# def recommendation_user_to_user():
#     user_id = request.args.get('user_id')
#     print(user_id)
#     try:
#         list_current_product_id = preProcessDataset1.get_user_current_products([user_id])

#         # similar products by users
#         dict_users_id_similarity = similarUsers.neighbors_user_id(user_id)
#         list_similar_users_product_id = preProcessDataset1. \
#             get_user_current_products(list(dict_users_id_similarity.keys()))
#         df_neigh_products = preProcessDataset3.get_products_information_by_id(list_similar_users_product_id)
#         # removing the products the user already has
#         df_neigh_products = df_neigh_products.loc[~df_neigh_products.ProdutoId.isin(list_current_product_id)]

#         mean_neighbors_risk = df_neigh_products.RiscoAtivo__c.mean()

#         # calculate similarity product user
#         similarity_dict = similarityEmbedding.cossine_distance([user_id], list_similar_users_product_id)

#         # add similarity
#         df_neigh_products = preProcessDataset3.add_similarity_column(df_neigh_products, similarity_dict)

#         # current information of the user products
#         df_current_products = preProcessDataset3.get_products_information_by_id(list_current_product_id)
#         mean_current_risk = df_current_products.RiscoAtivo__c.mean()

#         # current information about the user
#         df_user = preProcessDataset1.get_user_information_by_id([user_id])
#         user = from_dataframe_to_list_dict(df_user)
#         user[0].update({'mean_neighbors_risk': mean_neighbors_risk, 'mean_current_risk': mean_current_risk})

#         # how many products should the user buy
#         df_regression = quantityProductRegression.fit_transform(user_id, df_neigh_products)

#         list_products = from_dataframe_to_list_dict(df_regression)
#         return jsonify({"neighbor_products": list_products, "user": user})
#     except Exception as exc:
#         return "Error: {0}".format(exc)


# @app.route('/api/recommendations/product_to_product', methods=['POST', 'GET'])
# def recommendation_product_to_product():
#     user_id = request.args.get('user_id')
#     try:
#         list_current_product_id = preProcessDataset1.get_user_current_products([user_id])
#         print(list_current_product_id)
#         # similar products from products
#         dict_product_id_similarity = similarProducts.neighbors_product_idx(list_current_product_id, n_closest=5)
#         df_products = preProcessDataset3.get_products_information_by_id(dict_product_id_similarity.keys())
#         # removing the products the user already has
#         df_products = df_products.loc[~df_products.ProdutoId.isin(list_current_product_id)]
#         mean_recommended_risk = df_products.RiscoAtivo__c.mean()

#         # add similarity
#         df_products = preProcessDataset3.add_similarity_column(df_products, dict_product_id_similarity)

#         # current information of the user products
#         df_current_products = preProcessDataset3.get_products_information_by_id(list_current_product_id)
#         print(df_current_products)
#         mean_current_risk = df_current_products.RiscoAtivo__c.mean()

#         # current information about the user
#         df_user = preProcessDataset1.get_user_information_by_id([user_id])
#         user = from_dataframe_to_list_dict(df_user)
#         user[0].update({'mean_recommended_risk': mean_recommended_risk, 'mean_current_risk': mean_current_risk})

#         # how many products should the user buy
#         df_regression = quantityProductRegression.fit_transform(user_id, df_products)

#         list_products = from_dataframe_to_list_dict(df_regression)
#         return jsonify({"neighbor_products": list_products, "user": user})

#     except Exception as exc:
#         return "Error: {0}".format(exc)


if __name__ == '__main__':

    # try:
    #     similarityEmbedding = SimilarityEmbeddings()
    #     similarProductUsers = SimilarProductsUsers()
    #     similarUsers = SimilarUsers()
    #     similarProducts = SimilarProducts()
    #     preProcessDataset1 = PreProcessDataset1()
    #     preProcessDataset3 = PreProcessDataset3()
    #     quantityProductRegression = QuantityProductRegression()
    # except Exception as exc:
    #     print(exc)
    #     raise exc

    app.run(debug=False, host='0.0.0.0', port=5000)

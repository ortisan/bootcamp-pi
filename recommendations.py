import numpy as np
from daos import Users, Products
from helpers import normalized_dot_product
from models import RandomForestRegressor

rfr = RandomForestRegressor()


class SimilarUsers:
    def __init__(self):
        self.usersDao = Users()
        self.user_embeddings = self.usersDao.get_embeddings()

    def neighbors_user_id(self, user_id, n_closest=5):
        user_idx = self.userDao.user_id_to_idx(user_id)
        dists = np.dot(self.user_embeddings, self.user_embeddings[user_idx])
        closest_user_idx = np.argsort(dists)[-n_closest:]
        return closest_user_idx


class SimilarProducts:
    def __init__(self):
        self.productsDao = Products()
        self.product_embeddings = self.productsDao.get_embeddings()

    def neighbors_product_idx(self, product_idx, n_closest=5):
        list_user = []
        dists = np.dot(self.product_embeddings, self.product_embeddings[product_idx])
        closest_product_idx = np.argsort(dists)[-n_closest:]
        return closest_product_idx


class SimilarProductsUsers:
    def __init__(self):
        self.usersDao = Users()
        self.user_embeddings = self.usersDao.get_embeddings()        
        self.productsDao = Products()
        self.product_embeddings = self.productsDao.get_embeddings()

    def neighbors_products(self, user_id, n_closest=5):
        user_idx = self.usersDao.user_id_to_idx([user_id])
        dists = np.dot(self.product_embeddings, self.user_embeddings[user_idx])
        closest_product_idx = list(reversed(np.argsort(dists)[-n_closest:]))
        similarity_dict = {}
        for c in closest_product_idx:
            local_product_id = self.productsDao.idx_to_products([c])[['ProdutoId']].values[0][0]
            similarity_dict.update({local_product_id: dists[c]})
        return similarity_dict

    def neighbors_user_idx(self, product_idx, n_closest=5):
        dists = np.dot(self.user_embeddings, self.product_embeddings[product_idx])
        closest_user_idx = np.argsort(dists)[-n_closest:]
        return closest_user_idx


class QuantityProductRegression:
    def __init__(self):
        self.usersDao = Users()
        self.productDao = Products()

    def fit_transform(self, user_id, df_products):
        list_product_id = list(df_products.ProdutoId.values)
        user_embedding = self.usersDao.user_id_to_embedding([user_id])
        product_embeddings = self.productDao.product_id_to_embedding(list_product_id)
        df_dot_product = normalized_dot_product(user_embedding, product_embeddings)
        df_regression = rfr.predict(df_dot_product)

        dict_product_regression = {}
        for _, row in df_regression.iterrows():
            dict_product_regression.update({row['ProdutoId']: row['ProductQuantity']})

        df_products['ProductQuantity'] = df_products['ProdutoId'].map(dict_product_regression)
        return df_products






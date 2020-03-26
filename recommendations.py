import numpy as np
from daos import Users, Products, DotProductsUser


class SimilarUsers:
    def __init__(self):
        # self.user_embeddings = user_embeddings
        self.usersDao = Users()
        self.user_embeddings = self.usersDao.get_embeddings()


    def neighbors_user_id(self, user_id, n_closest=5):
        user_idx = self.userDao.user_id_to_idx(user_id)
        list_user = []
        dists = np.dot(self.user_embeddings, self.user_embeddings[user_idx])
        closest_user_idx = np.argsort(dists)[-n_closest:]
        return closest_user_idx


class SimilarProducts:
    def __init__(self):
        # self.product_embeddings = product_embeddings
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
        # self.user_embeddings = user_embeddings
        # self.product_embedding = product_embedding

    def neighbors_products(self, user_id, n_closest=5):
        list_user = []
        user_idx = self.usersDao.user_id_to_idx([user_id])
        dists = np.dot(self.product_embeddings, self.user_embeddings[user_idx])
        closest_product_idx = np.argsort(dists)[-n_closest:]
        return self.productsDao.idx_to_products(closest_product_idx.tolist())[['ProdutoId']].values.reshape(1, -1)[0]

    def neighbors_user_idx(self, product_idx, n_closest=5):
        list_user = []
        dists = np.dot(self.user_embeddings, self.product_embeddings[product_idx])
        closest_user_idx = np.argsort(dists)[-n_closest:]
        return closest_user_idx




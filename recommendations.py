import numpy as np
from daos import Users, Products, DotProductsUser


class SimilarUsers:
    def __init__(self):
        # self.user_embeddings = user_embeddings
        usersDao = Users()
        self.user_embeddings = usersDao.get_embeddings()


    def neighbors_user_idx(self, user_idx, n_closest=5):
        list_user = []
        dists = np.dot(self.user_embeddings, self.user_embeddings[user_idx])
        closest_user_idx = np.argsort(dists)[-n_closest:]
        return closest_user_idx


class SimilarProducts:
    def __init__(self):
        # self.product_embeddings = product_embeddings
        productsDao = Products()
        self.product_embeddings = productsDao.get_embeddings()

    def neighbors_product_idx(self, product_idx, n_closest=5):
        list_user = []
        dists = np.dot(self.product_embeddings, self.product_embeddings[product_idx])
        closest_product_idx = np.argsort(dists)[-n_closest:]
        return closest_product_idx


class SimilarProductsUsers:
    def __init__(self):
        usersDao = Users()
        self.user_embeddings = usersDao.get_embeddings()        
        productsDao = Products()

        self.product_embeddings = productsDao.get_embeddings()
        # self.user_embeddings = user_embeddings
        # self.product_embedding = product_embedding

    def neighbors_product_idx(self, user_idx, n_closest=5):
        list_user = []
        dists = np.dot(self.product_embeddings, self.user_embeddings[user_idx])
        closest_product_idx = np.argsort(dists)[-n_closest:]
        return closest_product_idx

    def neighbors_user_idx(self, product_idx, n_closest=5):
        list_user = []
        dists = np.dot(self.user_embeddings, self.product_embeddings[product_idx])
        closest_user_idx = np.argsort(dists)[-n_closest:]
        return closest_user_idx




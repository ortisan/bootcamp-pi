from keras.models import load_model
import _pickle as cPickle


class EmbeddingModel:
    def __init__(self):
        self.model = None

    def load(self):
        self.model = load_model('./model/recommender_embeddings.h5')
    
    def predict(self, encoded_product_id, encoded_user_id):
        prediction = self.model.predict([encoded_product_id, encoded_user_id])
        return prediction

    def get_model(self):
        return self.model

class RandomForestRegressor:
    def __init__(self):
        self.model = None
    
    def load(self):
        with open('./model/randomForestRegressor.cpickle', 'rb') as f:
            self.model = cPickle.load(f)

    def predict(self, dot_product_user):
            '''
            :param dot_product_user: o produto interno precisa ser normalizado antes da estimação
            no arquivo helpers tem uma função para normalizar o produto
            :return:
            '''
            return self.model.predict(dot_product_user)

    def get_model(self):
        return self.model
    


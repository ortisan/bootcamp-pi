import _pickle as cPickle
import math


class RandomForestRegressor:
    def __init__(self):
        self.model = None
        self.load()
    
    def load(self):
        with open('./models/randomForestRegressor.cpickle', 'rb') as f:
            self.model = cPickle.load(f)

    def predict(self, df_dot_product):
            '''
            :param df_dot_product: o produto interno precisa ser normalizado antes da estimação
            no arquivo helpers tem uma função para normalizar o produto
            :return:
            '''
            X = df_dot_product.iloc[:, 2:].values
            predictions = self.model.predict(X)
            df_regression = df_dot_product.iloc[:, :2]
            df_regression['ProductQuantity'] = predictions
            df_regression['ProductQuantity'] = df_regression['ProductQuantity']\
                .apply(lambda x: math.floor(x) if x > 1 else round(x))
            return df_regression

    def get_model(self):
        return self.model
    


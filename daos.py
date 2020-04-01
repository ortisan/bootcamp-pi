import pandas as pd


class Users:
    def __init__(self):
        self.dataframe = None
        self.load()
    
    def load(self):
        self.dataframe = pd.read_csv('./embeddings/userEncodedEmbedded.csv', index_col=0, low_memory=False)

    def user_id_to_encoding(self, list_user_id):
        df = self.dataframe.copy()
        return df.loc[df.userId.isin(list_user_id)]['encodUserId'].values
    
    def user_id_to_idx(self, user_id):
        df = self.dataframe.copy()
        return df.loc[df.userId == user_id].index[0]

    def idx_to_embedding(self, list_idx):
        df = self.dataframe.copy()
        return df.iloc[df.index.isin(list_idx)].values

    def user_id_to_embedding(self, list_user_id):
        df = self.dataframe.copy()
        df = df.drop(columns='encodUserId')
        return df.loc[df.userId.isin(list_user_id)]

    def idx_to_users(self, list_idx):
        df = self.dataframe.copy()
        df = df.iloc[df.index.isin(list_idx)]
        return df

    def get_embeddings(self):
        df = self.dataframe.copy()
        return df.iloc[:, 2:].values

    def get_users(self, list_user_id):
        df = self.dataframe.copy()
        return df.loc[df.userId.isin(list_user_id)]


class Products:
    def __init__(self):
        self.dataframe = None
        self.load()
    
    def load(self):
        self.dataframe = pd.read_csv('./embeddings/productEncodedEmbedded.csv', index_col=0, low_memory=False)

    def product_id_to_encoding(self, list_product_id):
        df = self.dataframe.copy()
        return df.loc[df.ProdutoId.isin(list_product_id)]['encodProdutoId'].values
    
    def product_id_to_idx(self, list_product_id):
        df = self.dataframe.copy()
        return df.loc[df.ProdutoId.isin(list_product_id)].index[0]

    def product_id_to_embedding(self, list_product_id):
        df = self.dataframe.copy()
        df = df.drop(columns='encodProdutoId')
        return df.loc[df.ProdutoId.isin(list_product_id)]

    def idx_to_embedding(self, list_idx):
        df = self.dataframe.copy()
        return df.iloc[df.index.isin(list_idx)].values

    def idx_to_products(self, list_idx):
        df = self.dataframe.copy()
        return df.iloc[df.index.isin(list_idx)]

    def get_embeddings(self):
        df = self.dataframe.copy()
        return df.iloc[:, 2:].values

    def get_products(self, list_product_id):
        df = self.dataframe.copy()
        return df.loc[df.ProdutoId.isin(list_product_id)]


class DotProductsUser:
    def __init__(self):
        self.dataframe = None
        self.load()

    def load(self):
        self.dataframe = pd.read_csv('./embeddings/dot_product_user.csv', index_col=0, low_memory=False)

    def user_id_to_product_id(self, list_user_id):
        df = self.dataframe.copy()
        return df.loc[df.UserId.isin(list_user_id)]['ProdutoId'].drop_duplicates().to_list()

    def product_id_to_user_id(self, list_product_id):
        df = self.dataframe.copy()
        return df.loc[df.ProdutoId.isin(list_product_id)]['UserId'].drop_duplicates().to_list()

    def get_product_user_embedding(self, product_id, user_id):
        df = self.dataframe.copy()
        return df.loc[(df.ProdutoId == product_id) & (df.UserId == user_id)]

    def get_embeddings(self):
        df = self.dataframe.copy()
        return df.iloc[:, 2:].values


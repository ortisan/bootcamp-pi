import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


class PreProcessDataset1:
    def __init__(self):
        self.dataFrame = None
        self.dataframe_processed = None
        self.load()
        self.process()

    def load(self):
        self.dataFrame = pd.read_csv('./datasets/Dataset-1.csv', encoding='utf_8', index_col=0, low_memory=False)

    def get_user_information_by_id(self, list_user_id):
        df = self.dataFrame.copy()
        df_users = df.loc[df.Id.isin(list_user_id)].drop_duplicates('Id')[['Id', 'Idade', 'NivelConhecimentoAtual', 'PerfilInvestidor', 'RendaMensal']]
        return df_users

    @staticmethod
    def add_similarity_column(df_users, dict_users_id_similarity):
        df_users['Similarity'] = df_users.Id.map(dict_users_id_similarity)
        df_users.sort_values('Similarity', ascending=False, inplace=True)
        return df_users

    def process(self):
        df_rating = self.dataFrame.copy()
        df_rating = df_rating.groupby(['Id', 'ProdutoId'])['Status'].count().reset_index()
        df_rating.sort_values('Id', inplace=True)
        df_rating.rename(columns={'Status': 'QtdProduto'}, inplace=True)

        df_rating = df_rating.pivot_table(index='Id', columns=['ProdutoId'],
                                          values='QtdProduto').reset_index().fillna(0)

        df_rating = df_rating.melt(id_vars='Id', value_name='QtdProduto')
        df_zeros = df_rating.loc[df_rating.QtdProduto == 0].sample(frac=0.9975, random_state=42)
        df_rating = df_rating.drop(df_zeros.index)
        df_rating.reset_index(inplace=True, drop=True)

        df_rating.QtdProduto = df_rating.QtdProduto.astype(int)
        df_rating['Link'] = np.where(df_rating.QtdProduto > 0, 1, 0)

        ordinal_encoder = OrdinalEncoder()

        user_id_encoded = ordinal_encoder.fit_transform(df_rating[['Id']])
        product_id_encoded = ordinal_encoder.fit_transform(df_rating[['ProdutoId']])

        df_rating_proc = pd.DataFrame(np.c_[df_rating.Id.values, df_rating.ProdutoId.values,
                                            user_id_encoded, product_id_encoded, df_rating.QtdProduto.values,
                                            df_rating.Link.values],
                                      columns=['userId', 'ProdutoId', 'encodUserId', 'encodProdutoId', 'QtdProduto',
                                               'Link'])

        self.dataframe_processed = df_rating_proc

    def get_user_current_products(self, list_user_id):
        df = self.dataframe_processed.copy()
        df = df.loc[df.QtdProduto > 0]
        return list(df.loc[df.userId.isin(list_user_id)]['ProdutoId'].values)


class PreProcessDataset3:
    def __init__(self):
        self.dataFrame = None
        self.load()

    def load(self):
        self.dataFrame = pd.read_csv('./datasets/Dataset-3.csv', encoding='utf_8', index_col=0, low_memory=False)

    def get_products_information_by_id(self, list_product_id):
        df = self.dataFrame
        df_product = df.loc[df.ProdutoId.isin(list_product_id)][['ProdutoId', 'DescricaoAtivo__c', 'RiscoAtivo__c']]
        return df_product

    @staticmethod
    def add_similarity_column(df_product, dict_products_id_similarity):
        df_product['Similarity'] = df_product.ProdutoId.map(dict_products_id_similarity)
        df_product.sort_values('Similarity', ascending=False, inplace=True)
        return df_product







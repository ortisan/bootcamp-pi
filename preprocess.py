import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


class PreProcessDataset1:
    def __init__(self):
        self.dataFrame = None

    def load(self):
        self.dataFrame = pd.read_csv('./datasets/Dataset-1.csv', encoding='utf_8', index_col=0)

    def get_user_information_by_id(self, list_user_id):
        self.load()
        df = self.dataFrame
        dict_list = list(df.loc[df.Id.isin([list_user_id])].drop_duplicates('Id')
                         [['Id', 'Idade', 'NivelConhecimentoAtual', 'PerfilInvestidor', 'RendaMensal']]
                         .T.to_dict().values())
        return dict_list

    def process(self):
        df_rating = self.dataFrame.groupby(['Id', 'ProdutoId'])['Status'].count().reset_index()
        df_rating.sort_values('Id', inplace=True)
        df_rating.rename(columns={'Status': 'QtdProduto'}, inplace=True)

        df_rating = df_rating.pivot_table(index='Id', columns=['ProdutoId'],
                                          values='QtdProduto').reset_index().fillna(0)

        df_rating = df_rating.melt(id_vars='Id', value_name='QtdProduto')
        df_zeros = df_rating.loc[df_rating.QtdProduto == 0].sample(frac=0.9975)
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

        return df_rating_proc


class PreProcessDataset3:
    def __init__(self):
        self.dataFrame = None

    def load(self):
        self.dataFrame = pd.read_csv('./datasets/Dataset-3.csv', encoding='utf_8', index_col=0)

    def get_products_information_by_id(self, list_product_id):
        self.load()
        df = self.dataFrame
        dict_list = list(df.loc[df.ProdutoId.isin(list_product_id)]
                         [['DescricaoAtivo__c', 'RiscoAtivo__c']].T.to_dict().values())
        return dict_list




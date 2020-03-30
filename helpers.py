import numpy as np
import pandas as pd


def normalized_dot_product(df_user_embeddings, df_product_embeddings):
    y = df_user_embeddings.iloc[:, 1:].values[0]
    user_id = df_user_embeddings.userId.values[0]
    list_aux = []
    for idx, row in df_product_embeddings.iterrows():
        x = row[1:].values
        w = x * np.dot(x, y) / np.dot(x, x)
        list_aux.append([user_id, row[0], list(w)])

    df_dot_product = pd.DataFrame(list_aux)
    df_dot_product.rename(columns={0: 'UserId', 1: 'ProdutoId', 2: 'dot_product_normalized'}, inplace=True)
    df_dot_embeddings = df_dot_product['dot_product_normalized'].apply(lambda s: pd.Series(s))
    df_dot_product = pd.concat([df_dot_product, df_dot_embeddings], axis=1)
    df_dot_product.drop(columns='dot_product_normalized', inplace=True)
    return df_dot_product


def from_dataframe_to_list_dict(dataframe):
    return list(dataframe.T.to_dict().values())

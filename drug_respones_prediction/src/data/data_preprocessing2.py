import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

def load_and_preprocess_data(ic50_path, drug_encoded_path, protein_path):
    # 读取数据
    ic50_data = pd.read_csv(ic50_path, header=0, index_col=0).stack().reset_index()
    ic50_data.columns = ['cell_line', 'drug_name', 'ic50']
    #print(ic50_data.head())

    # 处理药物编码
    drug_encoded_data = pd.read_csv(drug_encoded_path)
    drug_names = drug_encoded_data.columns.tolist()
    drug_encoded_data = drug_encoded_data[1:].reset_index(drop=True).T
    drug_encoded_data.columns = ['drug_name'] + [f'encoded_{i}' for i in range(1, len(drug_encoded_data.columns))]
    drug_encoded_data['drug_name'] = drug_names

    # 处理蛋白质数据
    protein_expression_data = pd.read_csv(protein_path, header=0, index_col=0).reset_index()
    protein_expression_data.columns = ['cell_line'] + [f'protein_{col}' for col in protein_expression_data.columns[1:]]
    #print(protein_expression_data.head())

    # 合并数据
    merged_data = pd.merge(
        pd.merge(ic50_data, drug_encoded_data, on='drug_name', how='inner'),
        protein_expression_data, on='cell_line', how='inner'
    )

    # 分离特征、目标和标识符
    identifiers = merged_data[['cell_line', 'drug_name']]  # 新增标识符
    #print(identifiers)
    X = merged_data.drop(['cell_line', 'drug_name', 'ic50'], axis=1).astype(np.float32)
    #print(X.head())
    #print(len(X.columns))
    y = merged_data['ic50']

    # PCA处理
    #X_sparse = csr_matrix(X)
    #print(X_sparse.shape)
    #pca = PCA(n_components=0.95)
    #X_pca = pca.fit_transform(X_sparse.toarray())
    #print(X_pca.shape)

    return X, y, identifiers  # 返回标识符
    #return X_pca, y, identifiers
#load_and_preprocess_data('filtered_ic50_data2.csv', 'one_hot_drugs.csv', 'DAE_features_dim200.csv')

import os

# 项目路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径
DATA_DIR = os.path.join(BASE_DIR, "data")
DRUG_GRAPHS_DIR = os.path.join(DATA_DIR, "drug_graphs")

# 输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型参数
HIDDEN_CHANNELS = 64
OUTPUT_CHANNELS = 32  # 最终特征向量维度
NUM_GCN_LAYERS = 3    # GCN层数
BATCH_SIZE = 32       # 批处理大小，对于单图也可适用

# 输出文件
DRUG_FEATURES_FILE = os.path.join(OUTPUT_DIR, "drug_features.csv")
FEATURE_VISUALIZATION_FILE = os.path.join(OUTPUT_DIR, "drug_features_visualization.png")
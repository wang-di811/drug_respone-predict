import os
import torch
import argparse
from src.feature_extractor import DrugFeatureExtractor
from src.utils import visualize_drug_features
import config

def main():
    parser = argparse.ArgumentParser(description='药物分子图GCN特征提取工具')
    parser.add_argument('--graphs_dir', type=str, default=config.DRUG_GRAPHS_DIR,
                        help='药物分子图目录路径')
    parser.add_argument('--output_file', type=str, default=config.DRUG_FEATURES_FILE,
                        help='输出特征CSV文件路径')
    parser.add_argument('--hidden_dim', type=int, default=config.HIDDEN_CHANNELS,
                        help='GCN隐藏层维度')
    parser.add_argument('--output_dim', type=int, default=config.OUTPUT_CHANNELS,
                        help='输出特征维度')
    parser.add_argument('--num_layers', type=int, default=config.NUM_GCN_LAYERS,
                        help='GCN层数')
    parser.add_argument('--visualize', action='store_true', help='是否可视化特征')
    args = parser.parse_args()
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查输入目录
    if not os.path.exists(args.graphs_dir):
        raise ValueError(f"药物分子图目录不存在: {args.graphs_dir}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 配置模型
    model_config = {
        'hidden_channels': args.hidden_dim,
        'out_channels': args.output_dim,
        'num_layers': args.num_layers
    }
    
    # 初始化特征提取器
    extractor = DrugFeatureExtractor(model_config)
    
    # 提取特征
    try:
        print("开始特征提取过程...")
        features_df = extractor.extract_features(args.graphs_dir, args.output_file)
        print(f"特征提取完成！结果已保存到: {args.output_file}")
        
        # 可视化特征
        if args.visualize:
            viz_file = config.FEATURE_VISUALIZATION_FILE
            print("生成特征可视化...")
            visualize_drug_features(features_df, viz_file, method='tsne')
            print(f"可视化已保存到: {viz_file}")
        
    except Exception as e:
        print(f"特征提取过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
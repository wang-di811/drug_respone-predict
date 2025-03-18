import os
import torch
import argparse
import yaml
import torch.nn as nn

# 导入自定义模块
from src.utils.logger import setup_logger
from src.data.data_loader import DataLoader
from src.data.cross_validation import CrossValidator
from src.training.cv_trainer import CVTrainer


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path):
    """主函数 - 交叉验证版本"""
    # 加载配置
    config = load_config(config_path)

    # 设置日志
    logger = setup_logger(config)
    logger.info("开始药物反应预测任务 - 五折交叉验证")

    # 检查CUDA可用性
    device_name = "cuda" if torch.cuda.is_available() and config['training']['use_cuda'] else "cpu"
    device = torch.device(device_name)
    logger.info(f"使用设备: {device}")

    # 数据加载
    data_loader = DataLoader(config)
    X, y, identifiers = data_loader.load_data()

    # 创建交叉验证器
    cv = CrossValidator(
        config,
        n_splits=5,  # 五折交叉验证
        shuffle=True,
        stratify_by='drug_name'  # 可以根据需要选择'cell_line'或'drug_name'进行分层
    )

    # 创建交叉验证数据集
    folds = cv.create_folds(X, y, identifiers)

    # 创建交叉验证训练器
    cv_trainer = CVTrainer(config, device)

    # 执行交叉验证训练和评估
    cv_results = cv_trainer.train_and_evaluate(folds)

    logger.info("药物反应预测任务完成 - 五折交叉验证")

    return cv_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="药物反应预测系统 - 交叉验证版本")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    args = parser.parse_args()

    main(args.config)
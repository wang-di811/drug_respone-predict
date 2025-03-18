import os
import torch
import argparse
import yaml
import torch.nn as nn

# 导入自定义模块
from src.utils.logger import setup_logger
from src.data.data_loader import DataLoader
from src.data.dataset import create_data_loaders
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.visualization import plot_training_history

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path):
    """主函数"""
    # 加载配置
    config = load_config(config_path)

    # 设置日志
    logger = setup_logger(config)
    logger.info("开始药物反应预测任务")

    # 检查CUDA可用性
    device_name = "cuda" if torch.cuda.is_available() and config['training']['use_cuda'] else "cpu"
    device = torch.device(device_name)
    logger.info(f"使用设备: {device}")

    # 数据加载和预处理
    data_loader = DataLoader(config)
    X, y, identifiers = data_loader.load_data()
    data_dict = data_loader.split_data(X, y, identifiers)
    data_dict = data_loader.preprocess_data(data_dict)

    # 创建数据加载器
    loaders = create_data_loaders(data_dict, batch_size=config['training']['batch_size'])

    # 创建模型
    input_dim = data_dict['train']['X'].shape[1]
    model = create_model(config, input_dim)

    # 训练模型
    trainer = Trainer(model, config, device)
    best_model, train_losses, val_losses = trainer.train(
        loaders['train'], loaders['val']
    )

    # 保存训练历史
    results_dir = config['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    plot_training_history(
        train_losses, val_losses,
        save_path=os.path.join(results_dir, 'training_history.png')
    )

    # 评估模型
    criterion = nn.MSELoss()
    evaluator = Evaluator(best_model, criterion, device, config)
    evaluation_results = evaluator.evaluate(loaders['test'], data_dict['test']['id'])

    logger.info("药物反应预测任务完成")

    return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="药物反应预测系统")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
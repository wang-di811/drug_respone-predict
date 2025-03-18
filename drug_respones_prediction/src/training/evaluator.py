import torch
import numpy as np
import pandas as pd
import os
import logging
from src.utils.visualization import plot_training_history, plot_predictions, plot_metrics_radar
from src.utils.metrics import calculate_all_metrics  # 导入新的指标计算模块

logger = logging.getLogger('DrugResponse.Evaluator')


class Evaluator:
    """模型评估器"""

    def __init__(self, model, criterion, device, config):
        """
        初始化评估器

        Args:
            model: 要评估的模型
            criterion: 损失函数
            device: 计算设备 (CPU/GPU)
            config: 评估配置
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.config = config

    def evaluate(self, test_loader, identifiers=None):
        """
        评估模型性能

        Args:
            test_loader: 测试数据加载器
            identifiers: 数据标识符

        Returns:
            包含评估指标的字典
        """
        logger.info("开始在测试集上评估模型")

        if self.model is None:
            logger.warning("评估器中没有模型，仅计算指标")
            return {}

        self.model = self.model.to(self.device)
        self.model.eval()
        test_loss = 0.0
        predictions = []
        actuals = []

        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item() * features.size(0)

                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(targets.cpu().numpy().flatten())

        test_loss = test_loss / len(test_loader.dataset)

        # 计算所有评估指标
        metrics = calculate_all_metrics(actuals, predictions)
        metrics['test_loss'] = test_loss
        metrics['predictions'] = predictions
        metrics['actuals'] = actuals

        # 记录评估结果
        logger.info(f"测试集损失: {test_loss:.4f}")
        logger.info(f"测试集 MAE: {metrics['mae']:.4f}")
        logger.info(f"测试集 RMSE: {metrics['rmse']:.4f}")
        logger.info(f"测试集 R²: {metrics['r2']:.4f}")
        logger.info(f"测试集 Pearson相关系数: {metrics['pearson']:.4f}")
        logger.info(f"测试集 Spearman等级相关系数: {metrics['spearman']:.4f}")
        logger.info(f"测试集 MAPE: {metrics['mape']:.2f}%")
        logger.info(f"测试集 一致性指数(CI): {metrics['ci']:.4f}")

        # 保存预测结果
        if identifiers is not None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            results_dir = self.config['output']['results_dir']
            os.makedirs(results_dir, exist_ok=True)

            final_results = pd.DataFrame({
                'cell_line': identifiers['cell_line'],
                'drug_name': identifiers['drug_name'],
                'actual_ic50': actuals,
                'predicted_ic50': predictions
            })

            results_path = os.path.join(results_dir, f"predictions_{timestamp}.csv")
            final_results.to_csv(results_path, index=False)
            logger.info(f"预测结果已保存到 {results_path}")

            # 生成可视化
            self.visualize_results(predictions, actuals, metrics, results_dir, timestamp)

        return metrics

    def visualize_results(self, predictions, actuals, metrics, results_dir, timestamp):
        """
        生成并保存可视化结果

        Args:
            predictions: 预测值
            actuals: 真实值
            metrics: 评估指标
            results_dir: 结果保存目录
            timestamp: 时间戳
        """
        logger.info("生成可视化结果")

        # 绘制预测散点图
        predictions_path = os.path.join(results_dir, f"predictions_scatter_{timestamp}.png")
        plot_predictions(predictions, actuals, save_path=predictions_path)

        # 绘制评估指标雷达图
        metrics_to_plot = {
            'R²': metrics['r2'],
            'Pearson': metrics['pearson'],
            'Spearman': metrics['spearman'],
            '1-RMSE': 1 - min(1, metrics['rmse'] / max(actuals)),  # 归一化并取反，使得值越大越好
            '1-MAE': 1 - min(1, metrics['mae'] / max(actuals)),  # 归一化并取反
            'CI': metrics['ci'],
            'Exp.Var': metrics['explained_variance']
        }

        radar_path = os.path.join(results_dir, f"metrics_radar_{timestamp}.png")
        plot_metrics_radar(metrics_to_plot, save_path=radar_path)

        logger.info(f"预测散点图已保存到 {predictions_path}")
        logger.info(f"指标雷达图已保存到 {radar_path}")

        # 保存所有指标到CSV
        metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()
                                   if not isinstance(v, list) and k != 'test_loss'})
        metrics_csv_path = os.path.join(results_dir, f"metrics_{timestamp}.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        logger.info(f"评估指标已保存到 {metrics_csv_path}")
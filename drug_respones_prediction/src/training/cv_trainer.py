import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.visualization import plot_training_history, plot_fold_metrics
from src.utils.metrics import calculate_all_metrics

logger = logging.getLogger('DrugResponse.CVTrainer')


class CVTrainer:
    """交叉验证训练器"""

    def __init__(self, config, device):
        """
        初始化交叉验证训练器

        Args:
            config: 配置对象
            device: 计算设备
        """
        self.config = config
        self.device = device
        self.criterion = nn.MSELoss()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(config['output']['results_dir'], f"cv_{self.timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)

    def train_and_evaluate(self, folds):
        """
        在K折上训练和评估模型

        Args:
            folds: 包含K折数据的列表

        Returns:
            各折评估结果
        """
        logger.info(f"开始{len(folds)}折交叉验证训练和评估")

        all_metrics = []
        all_train_losses = []
        all_val_losses = []

        # 保存全局结果
        all_predictions = []
        all_actuals = []
        all_identifiers = pd.DataFrame()

        # 在每一折上训练和评估
        for fold_idx, fold_data in enumerate(folds):
            logger.info(f"===== 第{fold_idx + 1}折 =====")

            # 创建该折的模型实例
            input_dim = fold_data['train']['X'].shape[1]
            model = create_model(self.config, input_dim)

            # 训练模型
            fold_trainer = Trainer(model, self.config, self.device)
            best_model, train_losses, val_losses = fold_trainer.train(
                fold_data['train']['loader'],
                fold_data['val']['loader']
            )

            # 保存训练历史
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)

            # 绘制该折的训练历史
            fold_history_path = os.path.join(self.results_dir, f"fold_{fold_idx + 1}_history.png")
            plot_training_history(
                train_losses, val_losses,
                save_path=fold_history_path,
                title=f"第{fold_idx + 1}折训练和验证损失"
            )

            # 评估模型
            evaluator = Evaluator(best_model, self.criterion, self.device, self.config)
            fold_metrics = evaluator.evaluate(fold_data['val']['loader'])

            # 收集该折的评估指标
            fold_metrics['fold'] = fold_idx + 1
            all_metrics.append(fold_metrics)

            # 收集预测结果
            fold_predictions = np.array(fold_metrics['predictions'])
            fold_actuals = np.array(fold_metrics['actuals'])

            # 合并标识符和预测结果
            fold_results = pd.DataFrame({
                'fold': fold_idx + 1,
                'predicted_ic50': fold_predictions,
                'actual_ic50': fold_actuals
            })

            # 添加标识符列
            for col in fold_data['val']['id'].columns:
                fold_results[col] = fold_data['val']['id'][col].reset_index(drop=True)

            all_identifiers = pd.concat([all_identifiers, fold_results], ignore_index=True)

            # 收集预测和实际值，用于全局评估
            all_predictions.extend(fold_predictions)
            all_actuals.extend(fold_actuals)

            # 保存模型
            model_path = os.path.join(self.results_dir, f"fold_{fold_idx + 1}_model.pth")
            best_model.save(model_path)
            logger.info(f"保存第{fold_idx + 1}折模型到 {model_path}")

        # 计算全局评估指标
        '''global_mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_actuals)))
        global_rmse = np.sqrt(np.mean(np.square(np.array(all_predictions) - np.array(all_actuals))))
        global_r2 = 1 - (np.sum(np.square(np.array(all_predictions) - np.array(all_actuals))) /
                         np.sum(np.square(np.array(all_actuals) - np.mean(all_actuals))))

        # 计算各折平均评估指标
        avg_mae = np.mean([m['mae'] for m in all_metrics])
        avg_rmse = np.mean([m['rmse'] for m in all_metrics])
        avg_r2 = np.mean([m['r2'] for m in all_metrics])'''
        # 计算全局评估指标
        global_metrics = calculate_all_metrics(all_actuals, all_predictions)

        # 计算各折平均评估指标
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            if metric not in ['predictions', 'actuals', 'fold']:
                avg_metrics[metric] = np.mean([m[metric] for m in all_metrics if metric in m])

        # 保存所有预测结果
        all_identifiers.to_csv(os.path.join(self.results_dir, "all_predictions.csv"), index=False)

        # 记录交叉验证结果
        # 记录交叉验证结果
        logger.info("\n========== 交叉验证结果 ==========")

        # 记录基础指标
        logger.info("\n----- 基础回归指标 -----")
        logger.info(f"MAE: 全局 {global_metrics['mae']:.4f} (各折平均: {avg_metrics['mae']:.4f})")
        logger.info(f"RMSE: 全局 {global_metrics['rmse']:.4f} (各折平均: {avg_metrics['rmse']:.4f})")
        logger.info(f"R²: 全局 {global_metrics['r2']:.4f} (各折平均: {avg_metrics['r2']:.4f})")

        # 记录相关性指标
        logger.info("\n----- 相关性指标 -----")
        logger.info(f"Pearson相关系数: 全局 {global_metrics['pearson']:.4f} (各折平均: {avg_metrics['pearson']:.4f})")
        logger.info(
            f"Spearman相关系数: 全局 {global_metrics['spearman']:.4f} (各折平均: {avg_metrics['spearman']:.4f})")
        logger.info(f"一致性指数(CI): 全局 {global_metrics['ci']:.4f} (各折平均: {avg_metrics['ci']:.4f})")

        # 记录其他指标
        logger.info("\n----- 其他指标 -----")
        logger.info(f"MAPE: 全局 {global_metrics['mape']:.2f}% (各折平均: {avg_metrics['mape']:.2f}%)")
        logger.info(
            f"中位数绝对误差: 全局 {global_metrics['median_ae']:.4f} (各折平均: {avg_metrics['median_ae']:.4f})")
        logger.info(
            f"解释方差得分: 全局 {global_metrics['explained_variance']:.4f} (各折平均: {avg_metrics['explained_variance']:.4f})")
        logger.info(f"最大误差: 全局 {global_metrics['max_error']:.4f} (各折平均: {avg_metrics['max_error']:.4f})")
        logger.info(f"平均偏差: 全局 {global_metrics['mean_bias']:.4f} (各折平均: {avg_metrics['mean_bias']:.4f})")
        logger.info("====================================")

        # 保存交叉验证指标
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(os.path.join(self.results_dir, "cv_metrics.csv"), index=False)

        # 绘制各折评估指标对比图
        plot_fold_metrics(all_metrics, save_path=os.path.join(self.results_dir, "fold_metrics.png"))

        # 可视化全局预测结果
        evaluator = Evaluator(None, None, None, self.config)
        evaluator.visualize_results(
            all_predictions, all_actuals,
            self.results_dir, "global"
        )

        '''return {
            'all_metrics': all_metrics,
            'global_metrics': {
                'mae': global_mae,
                'rmse': global_rmse,
                'r2': global_r2
            },
            'avg_metrics': {
                'mae': avg_mae,
                'rmse': avg_rmse,
                'r2': avg_r2
            },
            'all_train_losses': all_train_losses,
            'all_val_losses': all_val_losses
        }'''
        return {
            'all_metrics': all_metrics,
            'global_metrics': global_metrics,
            'avg_metrics': avg_metrics,
            'all_train_losses': all_train_losses,
            'all_val_losses': all_val_losses
        }

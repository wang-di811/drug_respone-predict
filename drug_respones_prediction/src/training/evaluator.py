import torch
import numpy as np
import pandas as pd
import os
import logging
import traceback
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
        
        try:
            if self.model is None:
                logger.warning("评估器中没有模型，仅计算指标")
                return {}

            logger.info(f"将模型移至设备: {self.device}")
            self.model = self.model.to(self.device)
            self.model.eval()
            test_loss = 0.0
            predictions = []
            actuals = []
            
            logger.info(f"开始处理测试数据，共{len(test_loader)}批次")
            batch_count = 0
            
            with torch.no_grad():
                for features, targets in test_loader:
                    batch_count += 1
                    logger.info(f"处理批次 {batch_count}/{len(test_loader)}")
                    
                    # 记录输入数据的形状
                    logger.info(f"特征形状: {features.shape}, 目标形状: {targets.shape}")
                    
                    features, targets = features.to(self.device), targets.to(self.device)
                    outputs = self.model(features)
                    
                    # 记录输出形状
                    logger.info(f"原始输出形状: {outputs.shape}")
                    
                    outputs_squeezed = outputs.squeeze()
                    logger.info(f"压缩后输出形状: {outputs_squeezed.shape}")
                    
                    # 确保输出维度匹配目标维度
                    if outputs_squeezed.dim() == 0 and targets.dim() == 0:  # 单个标量情况
                        outputs_squeezed = outputs_squeezed.unsqueeze(0)
                        targets = targets.unsqueeze(0)
                    
                    loss = self.criterion(outputs_squeezed, targets)
                    test_loss += loss.item() * features.size(0)
                    
                    # 安全地转换为numpy数组
                    cpu_outputs = outputs_squeezed.cpu()
                    cpu_targets = targets.cpu()
                    
                    # 记录转换后的数据信息
                    logger.info(f"CPU输出形状: {cpu_outputs.shape}, CPU目标形状: {cpu_targets.shape}")
                    
                    # 转换为numpy，并显式展平确保一致性
                    np_outputs = cpu_outputs.numpy()
                    np_targets = cpu_targets.numpy()
                    
                    if np_outputs.ndim > 1:
                        np_outputs = np_outputs.flatten()
                    if np_targets.ndim > 1:
                        np_targets = np_targets.flatten()
                    
                    predictions.extend(np_outputs)
                    actuals.extend(np_targets)

            test_loss = test_loss / len(test_loader.dataset)
            
            # 确保数据类型一致性
            predictions = np.array(predictions, dtype=np.float64)
            actuals = np.array(actuals, dtype=np.float64)
            
            # 检查是否有无效值
            invalid_preds = np.isnan(predictions).sum() + np.isinf(predictions).sum()
            invalid_actuals = np.isnan(actuals).sum() + np.isinf(actuals).sum()
            
            if invalid_preds > 0 or invalid_actuals > 0:
                logger.warning(f"检测到无效值！预测中: {invalid_preds}, 实际值中: {invalid_actuals}")
                # 过滤无效值
                valid_indices = np.logical_and(
                    np.logical_and(~np.isnan(predictions), ~np.isinf(predictions)),
                    np.logical_and(~np.isnan(actuals), ~np.isinf(actuals))
                )
                predictions = predictions[valid_indices]
                actuals = actuals[valid_indices]
                logger.info(f"过滤后数据大小: {len(predictions)}")
            
            logger.info(f"预测值范围: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
            logger.info(f"实际值范围: [{np.min(actuals):.4f}, {np.max(actuals):.4f}]")
            
            # 计算所有评估指标
            logger.info("开始计算评估指标...")
            metrics = calculate_all_metrics(actuals, predictions)
            metrics['test_loss'] = test_loss
            metrics['predictions'] = predictions.tolist()  # 转换为列表以便序列化
            metrics['actuals'] = actuals.tolist()

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
                try:
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    results_dir = self.config['output']['results_dir']
                    os.makedirs(results_dir, exist_ok=True)
                    
                    logger.info("创建预测结果DataFrame")
                    
                    # 确保identifiers长度与预测结果一致
                    id_length = len(identifiers['cell_line'])
                    pred_length = len(predictions)
                    
                    if id_length != pred_length:
                        logger.warning(f"标识符长度({id_length})与预测长度({pred_length})不匹配")
                        # 调整长度
                        min_length = min(id_length, pred_length)
                        cell_lines = identifiers['cell_line'][:min_length] if id_length > min_length else identifiers['cell_line']
                        drug_names = identifiers['drug_name'][:min_length] if id_length > min_length else identifiers['drug_name']
                        preds = predictions[:min_length] if pred_length > min_length else predictions
                        acts = actuals[:min_length] if pred_length > min_length else actuals
                        
                        final_results = pd.DataFrame({
                            'cell_line': cell_lines,
                            'drug_name': drug_names,
                            'actual_ic50': acts,
                            'predicted_ic50': preds
                        })
                    else:
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
                    logger.info("开始生成可视化结果...")
                    self.visualize_results(predictions, actuals, metrics, results_dir, timestamp)
                except Exception as e:
                    logger.error(f"保存预测结果时出错: {str(e)}")
                    logger.error(traceback.format_exc())

            return metrics
            
        except Exception as e:
            logger.error(f"评估过程中发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            # 返回一个最小的结果集，避免完全崩溃
            return {'error': str(e), 'test_loss': float('inf')}

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
        try:
            logger.info("生成可视化结果")

            # 确保使用numpy数组
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # 检查数据有效性
            if len(predictions) == 0 or len(actuals) == 0:
                logger.warning("预测或实际值为空，跳过可视化")
                return
                
            # 安全获取最大值，避免空数组错误
            max_actual = np.max(actuals) if len(actuals) > 0 else 1.0
            
            # 绘制预测散点图
            predictions_path = os.path.join(results_dir, f"predictions_scatter_{timestamp}.png")
            logger.info(f"创建散点图，保存到 {predictions_path}")
            plot_predictions(predictions, actuals, save_path=predictions_path)

            # 绘制评估指标雷达图
            metrics_to_plot = {
                'R²': metrics['r2'],
                'Pearson': metrics['pearson'],
                'Spearman': metrics['spearman'],
                '1-RMSE': 1 - min(1, metrics['rmse'] / max_actual),  # 归一化并取反，使得值越大越好
                '1-MAE': 1 - min(1, metrics['mae'] / max_actual),  # 归一化并取反
                'CI': metrics['ci'],
                'Exp.Var': metrics['explained_variance']
            }

            radar_path = os.path.join(results_dir, f"metrics_radar_{timestamp}.png")
            logger.info(f"创建雷达图，保存到 {radar_path}")
            plot_metrics_radar(metrics_to_plot, save_path=radar_path)

            logger.info(f"预测散点图已保存到 {predictions_path}")
            logger.info(f"指标雷达图已保存到 {radar_path}")

            # 保存所有指标到CSV
            metrics_to_save = {k: v for k, v in metrics.items() 
                              if not isinstance(v, list) and k != 'test_loss'}
            metrics_df = pd.DataFrame({k: [v] for k, v in metrics_to_save.items()})
            metrics_csv_path = os.path.join(results_dir, f"metrics_{timestamp}.csv")
            metrics_df.to_csv(metrics_csv_path, index=False)
            logger.info(f"评估指标已保存到 {metrics_csv_path}")
            
        except Exception as e:
            logger.error(f"可视化结果时出错: {str(e)}")
            logger.error(traceback.format_exc())
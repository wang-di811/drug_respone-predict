# 在visualization.py顶部添加
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging
import traceback
import sys
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .metrics import calculate_all_metrics, concordance_index

# 获取日志记录器
logger = logging.getLogger('DrugResponse.Visualization')

def plot_training_history(train_losses, val_losses, save_path='training_history.png', title='Train and Validation Loss'):
    """
    绘制训练和验证损失曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
        title: 图表标题
    """
    try:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"训练历史图表已保存至: {save_path}")
        return True
    except Exception as e:
        logger.error(f"绘制训练历史图表失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def plot_predictions(predictions, actuals, save_path='prediction_scatter.png'):
    """
    绘制预测值与真实值的散点图

    Args:
        predictions: 预测值列表
        actuals: 真实值列表
        save_path: 保存路径
    """
    try:
        # 数据类型转换和验证
        predictions = np.array(predictions, dtype=float)
        actuals = np.array(actuals, dtype=float)
        
        # 检查并移除无效值
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals) | 
                      np.isinf(predictions) | np.isinf(actuals))
        
        if np.sum(valid_mask) < len(valid_mask):
            logger.warning(f"发现{len(valid_mask) - np.sum(valid_mask)}个无效值，将被忽略")
            predictions = predictions[valid_mask]
            actuals = actuals[valid_mask]
            
        if len(predictions) == 0 or len(actuals) == 0:
            logger.error("没有有效的数据点用于绘制散点图")
            return False
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(actuals, predictions, alpha=0.5)

        # 添加完美预测线
        min_val = min(min(actuals), min(predictions))
        max_val = max(max(actuals), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        # 添加趋势线
        z = np.polyfit(actuals, predictions, 1)
        p = np.poly1d(z)
        plt.plot(sorted(actuals), p(sorted(actuals)), 'b-', label=f'Trend Line (y = {z[0]:.4f}x + {z[1]:.4f})')

        plt.xlabel('True IC50')
        plt.ylabel('Predicted IC50')
        plt.title('IC50 Prediction Results')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"预测散点图已保存至: {save_path}")
        return True
    except Exception as e:
        logger.error(f"绘制预测散点图失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def plot_regression_metrics(y_true, y_pred, save_path='regression_metrics.png'):
    """
    绘制回归指标柱状图

    Args:
        y_true: 真实值列表
        y_pred: 预测值列表
        save_path: 保存路径
    """
    try:
        # 数据类型转换和验证
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        
        # 检查并移除无效值
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | 
                      np.isinf(y_true) | np.isinf(y_pred))
        
        if np.sum(valid_mask) < len(valid_mask):
            logger.warning(f"发现{len(valid_mask) - np.sum(valid_mask)}个无效值，将被忽略")
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            
        if len(y_true) == 0 or len(y_pred) == 0:
            logger.error("没有有效的数据点用于计算指标")
            return False
            
        # 计算指标
        metrics = calculate_all_metrics(y_true, y_pred)
        
        # 选择要展示的指标
        metrics_to_show = {
            'MAE': metrics.get('mae', 0),
            'RMSE': metrics.get('rmse', 0),
            'R²': metrics.get('r2', 0),
            'Pearson': metrics.get('pearson', 0),
            'Spearman': metrics.get('spearman', 0)
        }
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        plt.figure(figsize=(10, 6))

        names = list(metrics_to_show.keys())
        values = list(metrics_to_show.values())

        plt.bar(names, values)
        plt.ylabel('Value')
        plt.title('Regression Evaluation Metrics')
        plt.xticks(rotation=0)
        plt.ylim(0, max(1.0, max(values) * 1.1))  # 确保合适的Y轴范围
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"回归指标图已保存至: {save_path}")
        return True
    except Exception as e:
        logger.error(f"绘制回归指标图失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def plot_fold_metrics(fold_metrics, save_path='fold_metrics.png'):
    """
    绘制各折评估指标对比图

    Args:
        fold_metrics: 包含各折评估指标的列表
        save_path: 保存路径
    """
    try:
        # 验证输入数据
        if not fold_metrics or not isinstance(fold_metrics, list):
            logger.error("无效的fold_metrics数据")
            return False
            
        # 将列表转换为DataFrame以便于处理
        metrics_df = pd.DataFrame(fold_metrics)
        
        # 确保必需的列存在
        required_columns = ['fold', 'mae', 'rmse', 'r2']
        missing_columns = [col for col in required_columns if col not in metrics_df.columns]
        if missing_columns:
            logger.error(f"fold_metrics中缺少必需的列: {missing_columns}")
            return False
            
        # 确保保存目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # 提取需要对比的指标
        metrics_to_compare = ['mae', 'rmse', 'r2']
        fold_numbers = metrics_df['fold'].values

        plt.figure(figsize=(12, 8))

        # 设置颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # 在图中为每个指标分配一个子图
        for i, metric in enumerate(metrics_to_compare):
            plt.subplot(1, 3, i + 1)
            plt.bar(fold_numbers, metrics_df[metric], color=colors[i])
            plt.axhline(y=metrics_df[metric].mean(), color='r', linestyle='-',
                        label=f'Average: {metrics_df[metric].mean():.4f}')
            plt.xlabel('Fold')
            plt.ylabel(metric.upper())
            plt.title(f'{metric.upper()} Comparison Across Folds')
            plt.grid(True, axis='y', alpha=0.3)
            plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"各折指标对比图已保存至: {save_path}")
        return True
    except Exception as e:
        logger.error(f"绘制各折指标对比图失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def plot_metrics_radar(metrics, save_path='metrics_radar.png', figsize=(10, 8)):
    """
    绘制评估指标雷达图

    Args:
        metrics: 要绘制的指标字典 (键为指标名称，值为指标值)
        save_path: 保存路径
        figsize: 图表大小
    """
    try:
        # 验证输入数据
        if not metrics or not isinstance(metrics, dict):
            logger.error("无效的metrics数据")
            return False
            
        # 过滤并标准化数据
        radar_metrics = {}
        for k, v in metrics.items():
            # 只包含可以标准化到0-1的指标
            if k in ['r2', 'pearson', 'spearman', 'ci', 'explained_variance']:
                # 如果是数值且在合理范围内
                if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):
                    # 标准化到0-1之间
                    if v < 0:
                        radar_metrics[k] = 0
                    elif v > 1:
                        radar_metrics[k] = 1
                    else:
                        radar_metrics[k] = v
                        
        if not radar_metrics:
            logger.error("没有可用于雷达图的有效指标")
            return False
            
        # 确保保存目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # 获取指标名称和值
        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())

        # 计算角度
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合雷达图

        # 添加值
        values += values[:1]  # 闭合值列表

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        # 绘制多边形
        ax.plot(angles, values, linewidth=2, linestyle='solid', label="Performance Metrics")
        ax.fill(angles, values, alpha=0.25)

        # 添加每个类别的标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # 设置雷达图刻度
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.set_ylim(0, 1)

        # 添加标题
        plt.title('Model Performance Evaluation', size=15, y=1.1)

        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"指标雷达图已保存至: {save_path}")
        return True
    except Exception as e:
        logger.error(f"绘制指标雷达图失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def visualize_cv_results(cv_results, save_dir, config=None, logger=None):
    """
    可视化交叉验证结果
    Args:
        cv_results: 交叉验证结果字典
        save_dir: 保存图表的目录
        config: 配置信息
        logger: 日志记录器
    """
    import os
    import logging
    import numpy as np
    
    if logger is None:
        logger = logging.getLogger('DrugResponse.Visualization')
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info("开始生成可视化结果...")
    logger.info(f"保存目录: {save_dir}")
    
    success_count = 0
    
    # 验证cv_results结构
    if 'fold_data' not in cv_results:
        logger.error("cv_results中缺少'fold_data'键")
        return
    
    logger.info(f"cv_results数据结构: {list(cv_results.keys())}")
    
    # 1. 收集和处理数据
    try:
        fold_metrics = []
        all_predictions = []
        all_actuals = []
        
        for fold_idx, fold_data in enumerate(cv_results['fold_data']):
            # 验证每个fold_data的结构
            required_keys = ['metrics', 'predictions', 'actuals']
            missing_keys = [key for key in required_keys if key not in fold_data]
            if missing_keys:
                logger.error(f"第{fold_idx}折数据缺少必要的键: {missing_keys}")
                continue
                
            logger.info(f"处理第{fold_idx+1}折数据, 数据键: {list(fold_data.keys())}")
                
            # 收集评估指标
            metrics = fold_data['metrics']
            metrics['fold'] = fold_idx + 1
            fold_metrics.append(metrics)
            
            # 收集预测值和实际值
            try:
                preds = fold_data['predictions']
                acts = fold_data['actuals']
                
                logger.info(f"第{fold_idx+1}折数据: predictions类型={type(preds)}, actuals类型={type(acts)}")
                logger.info(f"第{fold_idx+1}折样本数: {len(preds)}")
                
                # 确保数据可以被转换和扩展
                preds_list = preds.tolist() if isinstance(preds, np.ndarray) else list(preds)
                acts_list = acts.tolist() if isinstance(acts, np.ndarray) else list(acts)
                
                all_predictions.extend(preds_list)
                all_actuals.extend(acts_list)
                
                logger.info(f"累计收集的样本数: predictions={len(all_predictions)}, actuals={len(all_actuals)}")
            except Exception as e:
                logger.error(f"处理第{fold_idx+1}折预测数据时出错: {str(e)}")
                logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"收集数据时出错: {str(e)}")
        logger.error(traceback.format_exc())
    
    # 2. 绘制各折指标对比图
    if fold_metrics:
        fold_metrics_path = os.path.join(save_dir, 'fold_metrics.png')
        if plot_fold_metrics(fold_metrics, save_path=fold_metrics_path):
            success_count += 1
    else:
        logger.error("无法收集到任何折的评估指标，跳过绘制各折指标对比图")
    
    # 3. 绘制全局预测散点图
    if all_predictions and all_actuals:
        predictions_scatter_path = os.path.join(save_dir, 'predictions_scatter_global.png')
        if plot_predictions(all_predictions, all_actuals, save_path=predictions_scatter_path):
            success_count += 1
    else:
        logger.error("无法收集到预测结果和实际值，跳过绘制全局预测散点图")
    
    # 4. 绘制全局回归指标图
    if all_predictions and all_actuals:
        regression_metrics_path = os.path.join(save_dir, 'regression_metrics_global.png')
        if plot_regression_metrics(all_actuals, all_predictions, save_path=regression_metrics_path):
            success_count += 1
    
    # 5. 绘制全局指标雷达图
    try:
        global_metrics = cv_results.get('global_metrics', {})
        if global_metrics:
            logger.info(f"全局指标: {global_metrics}")
            metrics_radar_path = os.path.join(save_dir, 'metrics_radar_global.png')
            if plot_metrics_radar(global_metrics, save_path=metrics_radar_path):
                success_count += 1
        else:
            logger.warning("缺少全局评估指标，跳过雷达图生成")
    except Exception as e:
        logger.error(f"处理全局指标时出错: {str(e)}")
        logger.error(traceback.format_exc())
    
    logger.info(f"可视化结果生成完毕，成功生成{success_count}个图表")
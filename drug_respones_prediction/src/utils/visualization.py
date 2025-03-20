# 在visualization.py顶部添加
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging

def plot_training_history(train_losses, val_losses, save_path='training_history.png', title='训练和验证损失'):
    """
    绘制训练和验证损失曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_predictions(predictions, actuals, save_path='prediction_scatter.png'):
    """
    绘制预测值与真实值的散点图

    Args:
        predictions: 预测值列表
        actuals: 真实值列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)

    # 添加完美预测线
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='prefect prediction')

    # 添加趋势线
    z = np.polyfit(actuals, predictions, 1)
    p = np.poly1d(z)
    plt.plot(sorted(actuals), p(sorted(actuals)), 'b-', label=f'趋势线 (y = {z[0]:.4f}x + {z[1]:.4f})')

    plt.xlabel('true IC50')
    plt.ylabel('predict IC50')
    plt.title('IC50 predict result')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_regression_metrics(metrics, save_path='regression_metrics.png'):
    """
    绘制回归指标柱状图

    Args:
        metrics: 包含各种指标的字典
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))

    names = list(metrics.keys())
    values = list(metrics.values())

    plt.bar(names, values)
    plt.ylabel('Numerical')
    plt.title('Regression Evaluation Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_fold_metrics(fold_metrics, save_path='fold_metrics.png'):
    """
    绘制各折评估指标对比图

    Args:
        fold_metrics: 包含各折评估指标的列表
        save_path: 保存路径
    """
    # 将列表转换为DataFrame以便于处理
    metrics_df = pd.DataFrame(fold_metrics)

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
                    label=f'average: {metrics_df[metric].mean():.4f}')
        plt.xlabel('flod')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} Comparison across folds')
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metrics_radar(metrics, save_path='metrics_radar.png', figsize=(10, 8)):
    """
    绘制评估指标雷达图

    Args:
        metrics: 要绘制的指标字典 (键为指标名称，值为指标值)
        save_path: 保存路径
        figsize: 图表大小
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # 获取指标名称和值
        categories = list(metrics.keys())
        values = list(metrics.values())

        # 确保值在0到1之间，用于雷达图
        normalized_values = []
        for v in values:
            if v < 0:
                normalized_values.append(0)  # 负值设为0
            elif v > 1:
                normalized_values.append(1)  # 大于1的值设为1
            else:
                normalized_values.append(v)

        # 计算角度
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合雷达图

        # 添加值
        values = normalized_values
        values += values[:1]

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
        plt.title('Radar Chart for Model Performance Evaluation', size=15, y=1.1)

        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        import logging
        logger = logging.getLogger('DrugResponse.Visualization')
        logger.error(f"绘制雷达图失败: {str(e)}")

# 在文件末尾添加以下函数

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
    
    if logger is None:
        logger = logging.getLogger('DrugResponse.Visualization')
    
    try:
        logger.info("开始生成可视化结果...")
        
        # 1. 绘制各折指标对比图
        fold_metrics = []
        all_predictions = []
        all_actuals = []
        
        for fold_idx, fold_data in enumerate(cv_results['fold_data']):
            metrics = fold_data['metrics']
            metrics['fold'] = fold_idx + 1
            fold_metrics.append(metrics)
            
            # 收集所有预测值和实际值
            all_predictions.extend(fold_data['predictions'])
            all_actuals.extend(fold_data['actuals'])
        
        # 绘制各折指标对比图
        fold_metrics_path = os.path.join(save_dir, 'fold_metrics.png')
        plot_fold_metrics(fold_metrics, save_path=fold_metrics_path)
        logger.info(f"保存各折指标对比图: {fold_metrics_path}")
        
        # 2. 绘制全局预测散点图
        predictions_scatter_path = os.path.join(save_dir, 'predictions_scatter_global.png')
        plot_predictions(all_predictions, all_actuals, save_path=predictions_scatter_path)
        logger.info(f"保存全局预测散点图: {predictions_scatter_path}")
        
        # 3. 绘制全局指标雷达图
        # 计算全局指标
        global_metrics = cv_results.get('global_metrics', {})
        if global_metrics:
            metrics_radar_path = os.path.join(save_dir, 'metrics_radar_global.png')
            plot_metrics_radar(global_metrics, save_path=metrics_radar_path)
            logger.info(f"保存全局指标雷达图: {metrics_radar_path}")
        
        logger.info("所有可视化结果已生成完毕")
        
    except Exception as e:
        logger.error(f"生成可视化结果时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
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
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测')

    # 添加趋势线
    z = np.polyfit(actuals, predictions, 1)
    p = np.poly1d(z)
    plt.plot(sorted(actuals), p(sorted(actuals)), 'b-', label=f'趋势线 (y = {z[0]:.4f}x + {z[1]:.4f})')

    plt.xlabel('真实 IC50 值')
    plt.ylabel('预测 IC50 值')
    plt.title('IC50 预测结果')
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
    plt.ylabel('数值')
    plt.title('回归评估指标')
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
                    label=f'平均: {metrics_df[metric].mean():.4f}')
        plt.xlabel('折')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} 各折对比')
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
        ax.plot(angles, values, linewidth=2, linestyle='solid', label="性能指标")
        ax.fill(angles, values, alpha=0.25)

        # 添加每个类别的标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # 设置雷达图刻度
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.set_ylim(0, 1)

        # 添加标题
        plt.title('模型性能评估雷达图', size=15, y=1.1)

        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        import logging
        logger = logging.getLogger('DrugResponse.Visualization')
        logger.error(f"绘制雷达图失败: {str(e)}")
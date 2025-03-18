import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    explained_variance_score
)


def calculate_all_metrics(y_true, y_pred):
    """
    计算所有评估指标

    Args:
        y_true: 真实值数组
        y_pred: 预测值数组

    Returns:
        包含所有评估指标的字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 确保输入是一维数组
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()

    # 基础回归指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 添加的新指标
    # 1. 皮尔逊相关系数
    pearson_corr, p_value = pearsonr(y_true, y_pred)

    # 2. 斯皮尔曼等级相关系数
    spearman_corr, sp_p_value = spearmanr(y_true, y_pred)

    # 3. 平均绝对百分比误差
    # 防止除以零
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # 4. 中位数绝对误差
    median_ae = median_absolute_error(y_true, y_pred)

    # 5. 一致性指数
    ci = concordance_index(y_true, y_pred)

    # 6. 解释方差得分
    explained_variance = explained_variance_score(y_true, y_pred)

    # 7. 最大误差
    max_error = np.max(np.abs(y_true - y_pred))

    # 8. 平均偏差
    mean_bias = np.mean(y_pred - y_true)

    # 汇总所有指标
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'mape': mape,
        'median_ae': median_ae,
        'ci': ci,
        'explained_variance': explained_variance,
        'max_error': max_error,
        'mean_bias': mean_bias
    }

    return metrics


def concordance_index(y_true, y_pred):
    """
    计算一致性指数 (Concordance Index)

    Args:
        y_true: 真实值数组
        y_pred: 预测值数组

    Returns:
        一致性指数，范围[0, 1]
    """
    pairs = 0
    concordant = 0

    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:  # 排除平局
                pairs += 1
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                        (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant += 1

    if pairs == 0:
        return 0.5  # 没有有效对比时默认为0.5

    return concordant / pairs
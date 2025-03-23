from datetime import datetime
import os
import pandas as pd
import torch
import argparse
import yaml
import torch.nn as nn
import traceback
# 导入自定义模块
from src.utils.logger import setup_logger
from src.data.data_loader import DataLoader
from src.data.cross_validation import CrossValidator
from src.training.cv_trainer import CVTrainer
from src.utils.visualization import visualize_cv_results  # 添加此行
from src.utils import visualization
def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_predictions(cv_results, config, logger):
    """保存药物响应预测结果"""
    # 创建保存结果的目录
    results_dir = config['output']['results_dir']
    print("这是保存结果的目录",results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 收集所有折的预测结果
    all_predictions = []
    
    for fold_idx, fold_data in enumerate(cv_results['fold_data']):
        predictions = fold_data['predictions']
        print(predictions)
        actuals = fold_data['actuals']
        fold_identifiers = fold_data['identifiers']
        
        for i in range(len(predictions)):
            all_predictions.append({
                'fold': fold_idx + 1,
                'cell_line': fold_identifiers['cell_line'].iloc[i],
                'drug_name': fold_identifiers['drug_name'].iloc[i],
                'actual_ic50': actuals[i],
                'predicted_ic50': predictions[i]
            })
    
    # 转换为DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # 保存所有预测结果
    predictions_file = os.path.join(results_dir, f"drug_response_predictions_{timestamp}.csv")
    predictions_df.to_csv(predictions_file, index=False)
    
    # 计算每种药物的平均预测性能
    drug_predictions = predictions_df.groupby('drug_name').agg({
        'actual_ic50': ['mean', 'std', 'count'],
        'predicted_ic50': ['mean', 'std']
    })
    
    # 重命名列
    drug_predictions.columns = ['actual_mean_ic50', 'actual_std_ic50', 'sample_count', 
                               'predicted_mean_ic50', 'predicted_std_ic50']
    
    # 计算预测误差
    drug_predictions['mean_absolute_error'] = abs(drug_predictions['actual_mean_ic50'] - drug_predictions['predicted_mean_ic50'])
    
    # 保存药物预测汇总
    drug_predictions_file = os.path.join(results_dir, f"drug_predictions_summary_{timestamp}.csv")
    drug_predictions.to_csv(drug_predictions_file)
    
    logger.info(f"药物响应预测结果已保存至: {predictions_file}")
    logger.info(f"药物预测汇总已保存至: {drug_predictions_file}")
    
    return predictions_file, drug_predictions_file

def main(config_path):
    """主函数 - 交叉验证版本"""
    try:
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
        #data_loader = DataLoader(config, num_workers=4)
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
        logger.info(f"这是交叉验证数据集:{folds}")
        #print("这是交叉验证数据集",folds)

        # 创建交叉验证训练器
        cv_trainer = CVTrainer(config, device)
        logger.info("正在创建交叉验证数据集")

        # 执行交叉验证训练和评估
        cv_results = cv_trainer.train_and_evaluate(folds)
        logger.info(f"这是交叉验证训练和评估:{cv_results}")

        # 保存预测结果
        predictions_file, summary_file = save_predictions(cv_results, config, logger)
        logger.info(f"这是预测结果main_cv:{predictions_file, summary_file}")

        # 添加可视化代码
        if config.get('visualization', {}).get('enabled', True):
            # 获取保存结果的目录
            results_dir = os.path.dirname(predictions_file)
            # 调用可视化函数
            visualization.visualize_cv_results(cv_results, results_dir, config, logger)
        
        logger.info("药物反应预测任务完成 - 五折交叉验证")
        
        return cv_results, predictions_file, summary_file
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        print(traceback.format_exc())
        return None, None, None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="药物反应预测系统 - 交叉验证版本")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    args = parser.parse_args()

    main(args.config)
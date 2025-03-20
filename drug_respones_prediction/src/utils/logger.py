import logging
import os
from datetime import datetime

import logging
import os
from datetime import datetime

def setup_logger(config):
    """
    设置日志记录器
    Args:
        config: 包含日志配置的对象
    Returns:
        配置好的logger对象
    """
    log_dir = config['output']['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"drug_response_prediction_{timestamp}.log")

    # 创建一个自定义的logger
    logger = logging.getLogger("DrugResponse")
    logger.setLevel(logging.INFO)
    
    # 清除任何现有的处理程序（防止重复日志）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 添加指定编码的文件处理程序
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 添加控制台处理程序
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 不使用全局的basicConfig，而是使用自定义的处理程序
    # 这样可以精确控制每个处理程序的配置
    
    logger.info(f"日志设置完成，日志文件位置: {log_file}")

    return logger
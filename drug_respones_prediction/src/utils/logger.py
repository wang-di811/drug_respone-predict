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

    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger("DrugResponse")
    logger.info(f"日志设置完成，日志文件位置: {log_file}")

    return logger
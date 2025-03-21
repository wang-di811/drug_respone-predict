from .neural_network import DrugResponseNN
#from .models.neural_network import DrugResponseNN
import logging

logger = logging.getLogger('DrugResponse.ModelFactory')


def create_model(config, input_dim):
    """
    根据配置创建模型

    Args:
        config: 模型配置
        input_dim: 输入特征维度

    Returns:
        创建的模型实例
    """
    model_type = config['model']['type'].lower()

    if model_type == 'mlp':
        logger.info(f"创建MLP模型: 输入维度 {input_dim}, 隐藏层 {config['model']['hidden_dims']}")
        return DrugResponseNN(
            input_dim=input_dim,
            hidden_dims=config['model']['hidden_dims'],
            dropout_rate=config['model']['dropout_rate'],
            activation=config['model']['activation']
        )
    else:
        logger.error(f"不支持的模型类型: {model_type}")
        raise ValueError(f"不支持的模型类型: {model_type}")
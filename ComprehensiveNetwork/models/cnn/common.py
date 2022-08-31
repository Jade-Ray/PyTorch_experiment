from typing import Dict, Optional, Tuple, Union

from torch import nn

CONV_LAYERS = {
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
    'Conv': nn.Conv2d,}

NORM_LAYERS = {
    'BN': nn.BatchNorm2d,
    'LN': nn.LayerNorm,}

ACTIVATION_LAYERS = {act.__name__: act for act in 
                     [nn.ReLU, nn.LeakyReLU, nn.Sigmoid]}


def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer.
    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.
    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()
    
    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized layer type {layer_type}')
    else:
        conv_layer = CONV_LAYERS.get(layer_type)
        
    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


def build_norm_layer(cfg: Dict, num_features: int, postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.
    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.
    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    
    norm_layer = NORM_LAYERS.get(layer_type)
    name = layer_type.lower() + str(postfix)
    
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    layer = norm_layer(num_features, **cfg_)
    
    for param in layer.parameters():
        param.requires_grad = requires_grad
    
    return name, layer


def build_activation_layer(cfg: Dict) -> nn.Module:
    """Build activation layer.
    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
    Returns:
        nn.Module: Created activation layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()
    
    layer_type = cfg_.pop('type')
    activation_layer = ACTIVATION_LAYERS.get(layer_type)
    
    layer = activation_layer(**cfg_)
    
    return layer

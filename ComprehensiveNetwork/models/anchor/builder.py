from .anchor_generator import AnchorGenerator

ANCHOR_GENERATORS = {
    'AnchorGenerator': AnchorGenerator}


def build_anchor_generator(cfg, default_args=None):
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()
    
    layer_type = cfg_.pop('type')
    anchor_generator = ANCHOR_GENERATORS.get(layer_type)
    
    if default_args is not None:
        for name, value in default_args.items():
            cfg_.setdefault(name, value)
    
    layer = anchor_generator(**cfg_)
    
    return layer
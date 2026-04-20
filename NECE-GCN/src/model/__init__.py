from . import NECEGCN

__models = {
    'NECEGCN': NECEGCN,
}

def create(model_type, **kwargs):
    model_name = model_type.split('-')[0]
    return __models[model_name].create(model_type, **kwargs)

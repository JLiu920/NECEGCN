import logging
import numpy as np
from .graphs import Graph
from .ntu_feeder import NTU_Feeder, NTU_Location_Feeder

__data_args = {
    'ntu_mutual': {'class': 11, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu120_mutual': {'class': 26, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
}


def normalize_digraph(A):
    A = np.array(A)
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def create(dataset, **kwargs):
    g = Graph(dataset, **kwargs)
    try:
        data_args = __data_args[dataset]
        num_class = data_args['class']
    except:
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()

    A_e = normalize_digraph(g.adj_e)
    T_ne = normalize_digraph(g.V_T)
    A_angle = normalize_digraph(g.adj_angle)
    T_na = normalize_digraph(g.angle_T)
    feeders = {
        'train': data_args['feeder'](dataset=dataset, phase='train', connect_joint=g.connect_joint, **kwargs),
        'eval': data_args['feeder'](dataset=dataset, phase='eval', connect_joint=g.connect_joint, **kwargs),
    }
    data_shape = feeders['train'].datashape
    if 'ntu' in dataset:
        feeders.update({'location': NTU_Location_Feeder(data_shape)})
    return feeders, data_shape, num_class, g.A, g.parts, A_e, T_ne, A_angle, T_na

import os, logging, numpy as np
import torch
class Graph():
    def __init__(self, dataset, graph, labeling, num_person_out=1, max_hop=10, dilation=1, normalize=True,
                 threshold=0.2, **kwargs):
        self.dataset = dataset
        self.labeling = labeling
        self.graph = graph
        if labeling not in ['spatial', 'distance', 'zeros', 'ones', 'eye', 'pairwise0', 'pairwise1', 'geometric',
                            'user']:
            logging.info('')
            logging.error('Error: Do NOT exist this graph labeling: {}!'.format(self.labeling))
            raise ValueError()
        self.normalize = normalize
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_person_out = num_person_out
        self.threshold = threshold
        self.num_node, self.edge, self.connect_joint, self.parts, self.center = self._get_edge()
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset in ['ntu_mutual', 'ntu120_mutual']:
            if self.graph == 'mutual':
                num_node = 50
                neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                                  (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                                  (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                                  (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                                  (22, 23), (23, 8), (24, 25), (25, 12)] + \
                                 [(26, 27), (27, 46), (28, 46), (29, 28), (30, 46),
                                  (31, 30), (32, 31), (33, 32), (34, 46), (35, 34),
                                  (36, 35), (37, 36), (38, 26), (39, 38), (40, 39),
                                  (41, 40), (42, 26), (43, 42), (44, 43), (45, 44),
                                  (47, 48), (48, 33), (49, 50), (50, 37)] + \
                                 [(21, 46)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                connect_joint = np.array(
                    [1, 20, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 45, 7, 6, 11, 10, 0, 25, 45,
                     27, 45, 29, 30, 31, 45, 33, 34, 35, 25, 37, 38, 39, 25, 41, 42, 43, 26, 32, 31, 36, 35])
                parts = [
                    np.array([5, 6, 7, 8, 22, 23]) - 1,
                    np.array([5, 6, 7, 8, 22, 23]) + 25 - 1,
                    np.array([9, 10, 11, 12, 24, 25]) - 1,
                    np.array([9, 10, 11, 12, 24, 25]) + 25 - 1,
                    np.array([13, 14, 15, 16]) - 1,
                    np.array([13, 14, 15, 16]) + 25 - 1,
                    np.array([17, 18, 19, 20]) - 1,
                    np.array([17, 18, 19, 20]) + 25 - 1,
                    np.array([1, 2, 3, 4, 21]) - 1,
                    np.array([1, 2, 3, 4, 21]) + 25 - 1
                ]
                center = 21 - 1
        else:
            logging.info('')
            logging.error('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
            raise ValueError()
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts, center

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        self.oA = A
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        if self.labeling == 'user':
            connect = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                       (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                       (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                       (17, 1), (18, 17), (19, 18), (20, 19), (22, 8),
                       (23, 7), (24, 12), (25, 11)]
            connect = [(i - 1, j - 1) for (i, j) in connect]
            Adj = np.zeros((25, 25))
            for (i, j) in connect:
                Adj[i, j] = 1
                Adj[j, i] = 1
            A = np.zeros((50, 50))
            for i in range(2):
                A[i * 25:(i + 1) * 25, i * 25:(i + 1) * 25] = Adj
            A = A + np.eye(50)
            A[20, 20 + 25] = 1
            A[20 + 25, 20] = 1
            A[0, 25] = 1
            A[25, 0] = 1
            self.edges = []
            for i in range(50):
                for j in range(i + 1, 50):
                    if A[i][j] != 0:
                        self.edges.append((i, j))
            A = self._normalize_digraph(A)
            E = len(self.edges)
            self.V_T = torch.zeros(50, E)
            for e, (u, v) in enumerate(self.edges):
                self.V_T[u, e] = 1
                self.V_T[v, e] = 1
            self.adj_e = torch.zeros(E, E)
            for i in range(E):
                u_i, v_i = self.edges[i]
                for j in range(E):
                    u_j, v_j = self.edges[j]
                    if (u_i == u_j) or (u_i == v_j) or (v_i == u_j) or (v_i == v_j):
                        self.adj_e[i, j] = 1
            connections = [(0, 1, 20), (0, 12, 13), (0, 16, 17), (1, 0, 12), (1, 0, 16), (1, 20, 4),
                           (1, 20, 8), (2, 20, 4), (2, 20, 8), (3, 2, 20), (4, 5, 6), (5, 6, 7),
                           (6, 7, 21), (7, 6, 22), (8, 9, 10), (9, 10, 11), (10, 11, 23),
                           (11, 10, 24), (12, 0, 6), (12, 13, 14), (13, 14, 15), (16, 17, 18),
                           (17, 18, 19), (20, 4, 5), (20, 8, 9)]
            offset = 25
            connections_second_person = [(x + offset, y + offset, z + offset) for x, y, z in connections]
            all_connections = connections + connections_second_person
            n = len(all_connections)
            self.adj_angle = torch.zeros(n, n)

            def find_common_elements(t1, t2):
                return len(set(t1) & set(t2))
            for i in range(n):
                for j in range(i + 1, n):
                    if find_common_elements(all_connections[i], all_connections[j]) >= 2:
                        self.adj_angle[i][j] = 1
                        self.adj_angle[j][i] = 1
            self.adj_angle += torch.eye(n)

            E = len(all_connections)
            self.angle_T = torch.zeros(50, E)
            for e, (u, v, i) in enumerate(all_connections):
                self.angle_T[u, e] = 1
                self.angle_T[v, e] = 1
                self.angle_T[i, e] = 1
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD

    def build_edge_adj(self, adj_v_self):
        num_nodes = adj_v_self.shape[0]
        edges_self = []
        for i in range(num_nodes):
            edges_self.append((i, adj_v_self[i]))
        E_self = len(edges_self)
        T_self = np.zeros((num_nodes, E_self))
        for e, (u, v) in enumerate(edges_self):
            T_self[u, e] = 1
            T_self[v, e] = 1
        adj_e_self = np.zeros((E_self, E_self))
        for i in range(E_self):
            u_i, v_i = edges_self[i]
            for j in range(E_self):
                u_j, v_j = edges_self[j]
                if (u_i == u_j) or (u_i == v_j) or (v_i == u_j) or (v_i == v_j):
                    adj_e_self[i, j] = 1
        return adj_e_self, T_self

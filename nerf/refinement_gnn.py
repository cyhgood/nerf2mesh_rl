"""Two sparse message-passing layers for the Stage 1 shared policy."""
import math

import numpy as np
import torch
from torch import nn

from nerf.refinement_candidates import CandidatePolicy


POLICY_VERSION = 'face_gnn_two_layer_v1'
TEXTURE_POLICY_VERSION = 'face_gnn_two_layer_texture_v2'
RANK_POLICY_VERSION = 'face_gnn_texture_categorical_v3'
GRAPH_KEYS = ('node_features', 'edge_index', 'edge_features', 'edge_feature_valid', 'face_to_node', 'node_to_face')


def policy_graph_state(graph):
    """Copy the sampling graph into the trajectory; paths to NPZs are not required."""
    state = {key: np.asarray(graph[key]).copy() for key in GRAPH_KEYS}
    for key in ('texture_raw', 'texture_valid', 'texture_pixel_count'):
        if key in graph:
            state[key] = np.asarray(graph[key]).copy()
    state['edge_length_scale'] = float(np.sqrt(graph['metadata']['candidate_info']['normalization_scales'][0]))
    state['texture_feature_version'] = graph['metadata'].get('texture_feature_version')
    return state


class SparseFacePolicy(CandidatePolicy):
    """Shared policy. Budget is deliberately excluded from both graph layers."""
    def __init__(self):
        nn.Module.__init__(self)
        self.node_encoder = nn.Linear(9, 32)  # ten graph features minus budget
        self.message = nn.ModuleList([nn.Linear(36, 32), nn.Linear(36, 32)])
        self.update = nn.ModuleList([nn.Linear(64, 32), nn.Linear(64, 32)])
        self.action = nn.Sequential(nn.Linear(33, 32), nn.Tanh(), nn.Linear(32, 1))

    def encode(self, graph):
        x = torch.as_tensor(graph['node_features'], dtype=torch.float32)
        # index 6 is the current budget and cannot enter cached embeddings.
        x = torch.cat((x[:, :6], x[:, 7:]), dim=1)
        h = torch.tanh(self.node_encoder(x))
        edges = torch.as_tensor(graph['edge_index'], dtype=torch.long)
        values = torch.as_tensor(graph['edge_features'], dtype=torch.float32)
        valid = torch.as_tensor(graph['edge_feature_valid'], dtype=torch.bool)
        scale = max(float(graph['edge_length_scale']), 1e-12)
        attr = torch.stack((torch.log1p(values[:, 0].clamp(min=0) / scale).clamp(max=5),
                            values[:, 1] / math.pi, valid[:, 0].float(), valid[:, 1].float()), dim=1)
        for message, update in zip(self.message, self.update):
            source, dest = edges[0], edges[1]
            edge_messages = torch.tanh(message(torch.cat((h[source], attr), dim=1)))
            total = torch.zeros_like(h).index_add(0, dest, edge_messages)
            counts = torch.zeros((len(h), 1), dtype=h.dtype).index_add(
                0, dest, torch.ones((len(dest), 1), dtype=h.dtype))
            h = torch.tanh(update(torch.cat((h, total / counts.clamp(min=1)), dim=1)))
        return h

    def score(self, embeddings, budget):
        budget = torch.as_tensor(budget, dtype=embeddings.dtype).reshape(-1, 1)
        return self.action(torch.cat((embeddings, budget), dim=1)).squeeze(-1)

    def logits(self, graph, face_ids, budgets):
        ids = np.asarray(face_ids, dtype=np.int64)
        mapping = np.asarray(graph['face_to_node'], dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= len(mapping)) or np.any(mapping[ids] < 0):
            raise ValueError('Policy face IDs do not belong to this graph snapshot')
        nodes = mapping[ids]
        inverse = np.asarray(graph['node_to_face'], dtype=np.int64)
        if np.any(nodes >= len(inverse)) or not np.array_equal(inverse[nodes], ids):
            raise ValueError('Policy graph snapshot has inconsistent face/node mapping')
        return self.score(self.encode(graph)[torch.as_tensor(nodes, dtype=torch.long)], budgets)


class TextureFacePolicy(SparseFacePolicy):
    def __init__(self):
        super().__init__()
        self.node_encoder = nn.Linear(12, 32)

    def encode(self, graph):
        from nerf.refinement_texture import TEXTURE_FEATURE_VERSION
        if graph.get('texture_feature_version') != TEXTURE_FEATURE_VERSION or graph['node_features'].shape[1] != 13:
            raise ValueError('Texture GNN requires the matching Sobel graph feature version; retrain incompatible policies')
        return super().encode(graph)


class RankedTextureFacePolicy(TextureFacePolicy):
    def __init__(self):
        super().__init__()
        self.stop_head = nn.Linear(1, 1)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias, -2.)

    def stop_score(self, budget):
        return self.stop_head(torch.as_tensor(budget, dtype=torch.float32).reshape(1, 1)).reshape(())

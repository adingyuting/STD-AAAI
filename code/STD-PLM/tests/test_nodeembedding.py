import sys
from pathlib import Path
import numpy as np
import torch

# ensure src is on path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from model.model import NodeEmbedding


def test_setadj_padding():
    adj = np.eye(3, dtype=np.float32)
    emb = NodeEmbedding(adj_mx=adj, node_emb_dim=4, k=5)
    assert emb.lap_eigvec.shape == (3, 5)
    # forward pass returns tensor on default device
    out = emb()
    assert out.shape == (3, 4)

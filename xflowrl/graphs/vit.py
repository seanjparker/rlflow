import timeit

import torch
from vit_pytorch import ViT
import onnx
import taso as ts

def _attention(graph, input, heads):
    d_model = input.dim(1)
    assert input.dim(1) % heads == 0
    weights = list()
    for i in range(3):
        weights.append(graph.new_weight(dims=(d_model, d_model)))
    # compute query, key, value tensors
    q = graph.matmul(input, weights[0])
    k = graph.matmul(input, weights[1])
    v = graph.matmul(input, weights[2])
    # reshape query, key, value to multiple heads
    q = graph.reshape(q, shape=(64, 16, 64))
    k = graph.reshape(k, shape=(64, 16, 64))
    v = graph.reshape(v, shape=(64, 16, 64))
    # transpose query, key, value for batched matmul
    q = graph.transpose(q, perm=(1, 0, 2), shuffle=True)
    k = graph.transpose(k, perm=(1, 2, 0), shuffle=True)
    v = graph.transpose(v, perm=(1, 0, 2), shuffle=True)
    # perform matrix multiplications
    logits = graph.matmul(q, k)
    output = graph.matmul(logits, v)
    # transpose the output back
    output = graph.transpose(output, perm=(1, 0, 2), shuffle=True)
    output = graph.reshape(output, shape=(64, 1024))

    # a final linear layer
    linear = graph.new_weight(dims=(d_model, d_model))
    output = graph.matmul(output, linear)
    return output


def build_graph_vit():
    graph = ts.new_graph()
    graph_in = graph.new_input(dims=(64, 1024))
    t = graph_in
    for m in range(6):
        t = _attention(graph, t, 16)
    return graph


if __name__ == "__main__":
    # built_graph = build_graph_vit()
    # onnx_model = ts.export_onnx(built_graph)
    # onnx.save(onnx_model, 'vit_base.onnx')
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    dummy_input = torch.randn(1, 3, 256, 256)
    # torch.onnx.export(v, dummy_input, "vit_old.onnx", opset_version=12)
    #

    img = torch.randn(1, 3, 256, 256)
    import time
    start = time.time()
    for i in range(100):
        _ = v(img)  # (1, 1000)
    end = time.time()
    print((end-start)/100)



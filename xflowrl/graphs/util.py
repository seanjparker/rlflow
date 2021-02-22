import onnx
import taso as ts


def export_onnx(graph, file_name):
    onnx_model = ts.export_onnx(graph)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)

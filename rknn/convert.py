import numpy as np
from rknn.api import RKNN
import argparse

parser = argparse.ArgumentParser(description="Пример скрипта с аргументами")
parser.add_argument("--onnx", required=True, help="Путь к входному файлу")
args = parser.parse_args()

ONNX_MODEL = args.onnx
RKNN_MODEL = ONNX_MODEL.split(".")[0] + ".rknn"
QUANTIZE_ON = False
DATASET = "."
IMG_SIZE = 640
# Create RKNN object
rknn = RKNN(verbose=True)

# pre-process config
print('--> Config model')
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3568')
print('done')

# Load ONNX model
print('--> Loading model')
ret = rknn.load_onnx(model=ONNX_MODEL, input_size_list=[[3, 640, 640]])
if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')

# Export RKNN model
print('--> Export rknn model')
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print('Export rknn model failed!')
    exit(ret)
print('done')
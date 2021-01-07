import sys
sys.path.append('/home/matve/working/YOLOv6')

import torch

# Загрузим модель натренированную модель из .pt файла
checkpoint = torch.load('yolo6n.pt', map_location=torch.device('cpu'), weights_only=False)
model = checkpoint['model']
model = model.float()
model.eval()

dummy_input = torch.randn(16, 3, 640, 640, dtype=torch.float)

# Экспортируем в .onnx
torch.onnx.export(
    model,
    dummy_input,
    "yolov6n.onnx",
    export_params=True,
    opset_version=11,  # Настройте версию если нужно
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'])

print("Модель успешно экспортирована в onnx")

# ===============================================

import onnx
import onnxruntime as ort
import torch

# Загружаем onnx модель
onnx_model = onnx.load("yolov6n.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Тестируем модель
dummy_input = torch.randn(1, 3, 640, 640).numpy()
ort_session = ort.InferenceSession("yolov6n.onnx")
outputs = ort_session.run(None, {"image_arrays": dummy_input})
print(outputs[0])

# # ===============================================

from hailo_sdk_client import ClientRunner

onnx_path = "yolov6n.onnx"
onnx_model_name = "detector"
chosen_hw_arch = "hailo8"

# Инициализируем ClientRunner
runner = ClientRunner(hw_arch=chosen_hw_arch)

end_node_names = [
    "/model.22/cv2.0/cv2.0.2/Conv",
    "/model.22/cv3.0/cv3.0.2/Conv",
    "/model.22/cv2.1/cv2.1.2/Conv",
    "/model.22/cv3.1/cv3.1.2/Conv",
    "/model.22/cv2.2/cv2.2.2/Conv",
    "/model.22/cv3.2/cv3.2.2/Conv"

]
net_input_shapes = {"image_arrays": [1, 3, 640, 640]}  # уменьши batch до 1

try:
    hn, npz = runner.translate_onnx_model(
        onnx_path,
        onnx_model_name,
        end_node_names=end_node_names,
        net_input_shapes=net_input_shapes,
    )
    print("Model translation successful.")
except Exception as e:
    print(f"Error during model translation: {e}")
    raise

hailo_model_har_name = f"{onnx_model_name}.har"

try:
    runner.save_har(hailo_model_har_name)
    print(f"HAR file saved as: {hailo_model_har_name}")
except Exception as e:
    print(f"Error saving HAR file: {e}")

# ===============================================

from hailo_sdk_client import ClientRunner
har_path = "detector.har"
runner = ClientRunner(har=har_path)
from pprint import pprint

try:
    hn_dict = runner.get_hn()  # Or use runner._hn if get_hn() is unavailable
    print("Inspecting layers from HailoNet (OrderedDict):")

    for key, value in hn_dict.items():
        print(f"Key: {key}")
        pprint(value)
        print("\\n" + "="*80 + "\\n")

except Exception as e:
    print(f"Error while inspecting hn_dict: {e}")

# ===============================================

import numpy as np
from PIL import Image
import os

# Paths to directories and files
image_dir = 'calibration'

# File paths for saving calibration data
calibration_data_path = os.path.join("calibration_data.npy")
processed_data_path = os.path.join("processed_calibration_data.npy")

# Initialize an empty list for calibration data
calib_data = []

# Process all image files in the directory
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = Image.open(img_path).convert("RGB").resize((640, 640))
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        calib_data.append(img_array)    

# Convert the calibration data to a NumPy array
calib_data = np.array(calib_data)

# Save the normalized calibration data
np.save(calibration_data_path, calib_data)
print(f"Normalized calibration dataset saved with shape: {calib_data.shape} to {calibration_data_path}")

# Scale the normalized data back to [0, 255]
processed_calibration_data = calib_data * 255.0

# Save the processed calibration data
np.save(processed_data_path, processed_calibration_data)
print(f"Processed calibration dataset saved with shape: {processed_calibration_data.shape} to {processed_data_path}")

# ===============================================

# import json
# import os

# nms_layer_config = {
#     "nms_scores_th": 0.2,
#     "nms_iou_th": 0.7,
#     "image_dims": [640, 640],
#     "max_proposals_per_class": 100,
#     "classes": 80,
#     "regression_length": 16,
#     "background_removal": False,
#     "background_removal_index": 0,
#     "bbox_decoders": [
#         {
#             "name": "bbox_decoder_8",
#             "stride": 8,
#             "reg_layer": "detector/conv77",
#             "cls_layer": "detector/conv76"
#         },
#         {
#             "name": "bbox_decoder_16",
#             "stride": 16,
#             "reg_layer": "detector/conv91",
#             "cls_layer": "detector/conv90"
#         },
#         {
#             "name": "bbox_decoder_32",
#             "stride": 32,
#             "reg_layer": "detector/conv104",
#             "cls_layer": "detector/conv103"
#         }
#     ]
# }

# # Save the JSON
# output_path = "nms_layer_config.json"
# with open(output_path, "w") as json_file:
#     json.dump(nms_layer_config, json_file, indent=4)

# print(f"NMS layer configuration saved to {output_path}")

# # ===============================================

import os
from hailo_sdk_client import ClientRunner

# Define your model's HAR file name
model_name = "detector"
hailo_model_har_name = f"{model_name}.har"

# Ensure the HAR file exists
assert os.path.isfile(hailo_model_har_name), "Please provide a valid path for the HAR file"

# Initialize the ClientRunner with the HAR file
runner = ClientRunner(har=hailo_model_har_name)

# Define the model script to add a normalization layer
# Normalization for [0, 1] range
alls = [

'norm_layer1 = normalization([0.0,0.0,0.0], [255.0,255.0,255.0])\n'
'resize_input1 = resize(resize_shapes=[480,640],engine=nn_core)\n'
# 'rgb1 = input_conversion(yuv_to_rgb)\n'
# 'yuy2_to_yuv1 = input_conversion(input_layer1, yuy2_to_hailo_yuv)\n'
"nms_postprocess( meta_arch=yolov6, engine=nn_core)\n",
]

# Load the model script into the ClientRunner
runner.load_model_script("".join(alls))

# Define a calibration dataset
# Replace 'calib_dataset' with the actual dataset you're using for calibration
# For example, if it's a directory of images, prepare the dataset accordingly
calib_dataset = "calib_set_480x640_rgb.npy"

# Perform optimization with the calibration dataset
runner.optimize(calib_dataset)

# Save the optimized model to a new Quantized HAR file
quantized_model_har_path = f"{model_name}_640_rgb_quantized_model.har"
runner.save_har(quantized_model_har_path)

print(f"Quantized HAR file saved to: {quantized_model_har_path}")

# # # # # # ===============================================

from hailo_sdk_client import ClientRunner

# Define the quantized model HAR file
model_name = "detector"
quantized_model_har_path = f"{model_name}_640_rgb_quantized_model.har"

# Initialize the ClientRunner with the HAR file
runner = ClientRunner(har=quantized_model_har_path)
print("[info] ClientRunner initialized successfully.")

# Compile the model
try:
    hef = runner.compile()
    print("[info] Compilation completed successfully.")
except Exception as e:
    print(f"[error] Failed to compile the model: {e}")
    raise
file_name = f"{model_name}.hef"
with open(file_name, "wb") as f:
    f.write(hef)

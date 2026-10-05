import sys

try:
    import torch
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
        print("Device Count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}:", torch.cuda.get_device_name(i))
    else:
        print("ERROR: CUDA is NOT available to PyTorch.")
except ImportError:
    print("ERROR: PyTorch is not installed.")

print("-" * 40)

try:
    from ultralytics import YOLO
    from ultralytics.utils.checks import check_yolo
    print("Checking Ultralytics YOLO GPU support:")
    check_yolo()
except ImportError:
    print("ERROR: ultralytics is not installed.")


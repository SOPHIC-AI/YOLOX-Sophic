## YOLOX-ONNXRuntime in Python

This doc introduces how to convert your pytorch model into onnx, and how to run an onnxruntime demo to verify your convertion.

### Convert Your Model to ONNX

First, you should move to <YOLOX_HOME> by:
```shell
cd <YOLOX_HOME>
```
Then, you can:
```shell
python3 tools/export_onnx.py --output-name your_yolox.onnx -f exps/example/yolox_voc/yolox_voc_nano.py -c {path of checkpoint}
```

### ONNXRuntime Demo

Step1.
```shell
cd <YOLOX_HOME>/demo/ONNXRuntime
```

Step2. 
```shell
python3 onnx_inference.py -m <ONNX_MODEL_PATH> -i <IMAGE_PATH> -o <OUTPUT_DIR> -s 0.3 --input_shape 640,640
```
Notes:
* -m: your converted onnx model
* -i: input_image
* -s: score threshold for visualization.
* --input_shape: should be consistent with the shape you used for onnx convertion.

# tensorrt_custom_layer_dynamic_batchnms

## Introduction
tensorrt custom layer for dynamic batchsize mns
## Quick start
1. Pull tensorrt docker image
```bashrc
$ docker pull yingchao126/tensorrt:20.03-py3
```

2. In the container clone this
```bashrc
$ git clone https://github.com/zhouyingchaoAI/tensorrt_custom_layer_dynamic_batchnms.git
```

3. Build the batchnms_plugin
```bashrc
$ cd tensorrt_custom_layer_dynamic_batchnms
$ mkdir build
$ cmake .. && make
```
4. put the libNMSPlugin.so to triton-inference-server container and commit new image
```bashrc
$ docker pull docker pull yingchao126/tensorrtserver:20.02-py3
$ docker run -it --rm --gpus device=0 --shm-size=4g --ulimit memlock=-1 --name trtserver --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002  yingchao126/tensorrtserver:20.02-py3  trtserver --model-repository=/models --pinned-memory-pool-byte-size=0 --tf-gpu-memory-fraction=0.5 2>&1
$ docker cp libNMSPlugin.so trtserver:/opt/tensorrtserver/lib
$ docker commit trtserver yingchao126/trtis_plugin:20.02-py3-trt-customlayer-dynamic-nms

5. run the new triton-inference-server with custom layer batchnms_plugin
$ docker run -it --rm --gpus device=0 --shm-size=4g --ulimit memlock=-1 --name trtserver --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v /your-model-path/:/models -e LD_PRELOAD=/opt/tensorrtserver/lib/libNMSPlugin.so yingchao126/trtis_plugin:20.02-py3-trt-customlayer-dynamic-nms  trtserver --model-repository=/models --pinned-memory-pool-byte-size=0 --tf-gpu-memory-fraction=0.5 2>&1
```
## Other Usefull Links

[- **`triton-inference-server`**](https://github.com/NVIDIA/triton-inference-server)<br>

[- **`onnx_tflite_yolov3`**](https://github.com/zldrobit/onnx_tflite_yolov3)

[- **`triton-inference-server docs`**](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/quickstart.html)

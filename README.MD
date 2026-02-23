# License Plate Detector

This license plate detector is a technology demonstration of how to use ONNX under Linux.

## Training

I annotated license plate images (4000+ jpeg files) and trained my model using YOLOv8.

- Long plates
+ Square(ish) plates
* Foreign plates

In order to load and use ONNX runtime under Linux:
1. Download ONNX Runtime from: https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime
2. Extract it (it is basically a zip file) to a folder
3. Go to `runtimes/linux-x64/native`
4. Copy `libonnxruntime.so` to `./onnx/lib` folder of your project
5. ln -s libonnxruntime.so libonnxruntime.so.1
6. Copy include files (recursively) to your project file as well
    
    `cp -r build/native/include/ ./onnx/`

### OpenCV2 folder name fix for Ubuntu 24.04

    sudo ln -s /usr/include/opencv4/opencv2 /usr/include/opencv2

Some *italic* and **bold** text and `inline code`.

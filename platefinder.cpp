#include "platefinder.h"
#include <sstream>
#include <iostream>
using std::cout;
using std::endl;

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<int64_t>& v) {
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

PlateFinder::PlateFinder(QString modelName)
{
    // onnxruntime setup
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Default");
    session_options = new Ort::SessionOptions();
    session_options->SetInterOpNumThreads(1);
    session_options->SetIntraOpNumThreads(1);
    session = new Ort::Session(*env, modelName.toStdString().c_str(), *session_options);

    // print name/shape of inputs
    std::vector<std::string> input_names = session->GetInputNames();
    cout << "Input Node Name/Shape (" << input_names.size() << "):" << endl;
    for (size_t i = 0; i < input_names.size(); i++) {
        cout << "\t" << input_names[i] << " : " << endl;
    }

    // print name/shape of outputs
    std::vector<std::string> output_names = session->GetOutputNames();
    cout << "Output Node Name/Shape (" << output_names.size() << "):" << endl;
    for (size_t i = 0; i < output_names.size(); i++) {
        cout << "\t" << output_names[i] << " : " << endl;
        auto output_shapes = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        for (size_t j = 0; j < output_shapes.size(); j++)
        {
            cout << output_shapes.at(j) << endl;
        }
    }

    // Assume model has 1 input node and 1 output node.
    assert(input_names.size() == 1 && output_names.size() == 1);

    auto providers = Ort::GetAvailableProviders();
    for (auto provider : providers) {
        std::cout << provider << std::endl;
    }
}

std::vector<PlateResult> PlateFinder::InspectPicture(cv::Mat &picture, float threshold)
{
    const int MODEL_WIDTH = 640;
    const int MODEL_HEIGHT = 640;

    cv::Mat resizedImage;
    cv::resize(picture, resizedImage, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));

    // OpenCV loads as BGR, but models usually expect RGB
    //cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);

    // Normalize pixel values from [0, 255] to [0.0, 1.0] and convert to float
    resizedImage.convertTo(resizedImage, CV_32FC3, 1.0f / 255.0f);

    // AI Models expect images in "NCHW" format (Batch, Channels, Height, Width)
    // But OpenCV stores them as "HWC" (Height, Width, Channels interleaved).
    // We must split the image into 3 separate red, green, and blue arrays.
    std::vector<float> inputTensorValues(1 * 3 * MODEL_HEIGHT * MODEL_WIDTH);

    std::vector<cv::Mat> chw(3);
    chw[0] = cv::Mat(cv::Size(MODEL_WIDTH, MODEL_HEIGHT), CV_32FC1, inputTensorValues.data()); // Red
    chw[1] = cv::Mat(cv::Size(MODEL_WIDTH, MODEL_HEIGHT), CV_32FC1, inputTensorValues.data() + MODEL_WIDTH * MODEL_HEIGHT); // Green
    chw[2] = cv::Mat(cv::Size(MODEL_WIDTH, MODEL_HEIGHT), CV_32FC1, inputTensorValues.data() + 2 * MODEL_WIDTH * MODEL_HEIGHT); // Blue
    cv::split(resizedImage, chw);

    // ---------------------------------------------------------
    // 3. CREATE ONNX TENSOR
    // ---------------------------------------------------------
    std::vector<int64_t> inputDims = {1, 3, MODEL_HEIGHT, MODEL_WIDTH};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorValues.size(),
        inputDims.data(),
        inputDims.size()
        );
    std::vector<PlateResult> results;
    // std::vector<Ort::Value> inputTensor;        // Onnxruntime allowed input

    // // this will make the input into 1,3,640,640
    // cv::Mat blob = cv::dnn::blobFromImage(picture, 1 / 255.0, cv::Size(640, 640), (0, 0, 0), false, false);
    // size_t input_tensor_size = blob.total();
    // try {
    //     inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, input_tensor_size, input_node_dims[0].data(), input_node_dims[0].size()));
    // }
    // catch (Ort::Exception oe) {
    //     std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
    //     return results;
    // }

    // ---------------------------------------------------------
    // 4. RUN INFERENCE
    // ---------------------------------------------------------
    // IMPORTANT: You must use the exact input and output node names from your model!
    // Common YOLO names are "images" and "output0".
    const char* inputNames[] = {"images"};
    const char* outputNames[] = {"output0"};

    auto outputTensors = session->Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor, 1,
        outputNames, 1
        );

    // ---------------------------------------------------------
    // 5. READ THE OUTPUT (POST-PROCESSING)
    // ---------------------------------------------------------
    float* outputData = outputTensors[0].GetTensorMutableData<float>();

    int rows = 300;
    int dimensions = 6;

    // Calculate scale factors (to stretch boxes to original image size)
    float x_factor = (float)picture.cols / MODEL_WIDTH;
    float y_factor = (float)picture.rows / MODEL_HEIGHT;

    // Loop through the 300 predictions
    for (int i = 0; i < rows; ++i) {
        // Point to the beginning of the current row
        float* rowPtr = outputData + (i * dimensions);

        // Modern NMS-Free YOLO models output this exact format:
        // rowPtr[0] = x_min
        // rowPtr[1] = y_min
        // rowPtr[2] = x_max
        // rowPtr[3] = y_max
        // rowPtr[4] = confidence score
        // rowPtr[5] = class ID (e.g., 0.0 for Class 0, 1.0 for Class 1)

        float score = rowPtr[4];

        // Filter out weak predictions
        if (score > 0.45f) { // Confidence threshold
            float x1 = rowPtr[0];
            float y1 = rowPtr[1];
            float x2 = rowPtr[2];
            float y2 = rowPtr[3];
            int classId = static_cast<int>(rowPtr[5]);

            // Scale the coordinates back to the original image size
            int left = int(x1 * x_factor);
            int top = int(y1 * y_factor);
            int right = int(x2 * x_factor);
            int bottom = int(y2 * y_factor);

            // Calculate width and height for OpenCV's Rect
            int width = right - left;
            int height = bottom - top;

            cv::Rect box(left, top, width, height);

            // Draw the rectangle
            cv::rectangle(picture, box, cv::Scalar(0, 255, 0), 3);

            // Create a label with the class ID and confidence score
            std::string label = "Class " + std::to_string(classId) + " (" + std::to_string(score).substr(0, 4) + ")";

            // Draw the text
            cv::putText(picture, label, cv::Point(box.x, box.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        }
    }

    // Display the result!
    cv::imshow("Plate Detection Result", picture);
    //cv::waitKey(0); // Wait until the user presses a key
    return results;
}

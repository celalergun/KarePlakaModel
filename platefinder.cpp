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
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);

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
    // Get a pointer to the raw float data coming out of the model
    // ---------------------------------------------------------
    // 5. READ THE OUTPUT (POST-PROCESSING)
    // ---------------------------------------------------------
    float* outputData = outputTensors[0].GetTensorMutableData<float>();

    // 1. Wrap the raw data in an OpenCV Mat and transpose it.
    // This changes [6, 8400] into [8400, 6], making it 8400 rows and 6 columns.
    int dimensions = 6;
    int rows = 8400;
    cv::Mat outputMat(dimensions, rows, CV_32F, outputData);
    cv::Mat transposedMat = outputMat.t();

    // Vectors to hold our filtered results
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    // 2. Calculate scale factors
    // The model predicted boxes for a 640x640 image, but we want to draw
    // them on the ORIGINAL image size.
    float x_factor = (float)picture.cols / MODEL_WIDTH;
    float y_factor = (float)picture.rows / MODEL_HEIGHT;

    // 3. Iterate through all 8400 rows
    for (int i = 0; i < rows; ++i) {
        float* rowPtr = transposedMat.ptr<float>(i);

        // rowPtr[0] = center_x, rowPtr[1] = center_y
        // rowPtr[2] = width,    rowPtr[3] = height
        // rowPtr[4] = class 0 score, rowPtr[5] = class 1 score

        // Find which class has the highest score
        float maxScore = -1.0f;
        int bestClassId = -1;

        for (int c = 4; c < dimensions; ++c) {
            if (rowPtr[c] > maxScore) {
                maxScore = rowPtr[c];
                bestClassId = c - 4; // Will result in 0 or 1
            }
        }

        // 4. Filter out weak predictions (Confidence Threshold)
        if (maxScore > 0.1f) { // You can adjust this threshold between 0.1 and 0.9
            float cx = rowPtr[0];
            float cy = rowPtr[1];
            float w = rowPtr[2];
            float h = rowPtr[3];

            // Convert center coordinates to top-left coordinates and scale them
            int left = int((cx - 0.5 * w) * x_factor);
            int top = int((cy - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(maxScore);
            classIds.push_back(bestClassId);
        }
    }

    // 5. Non-Maximum Suppression (NMS)
    // YOLO predicts multiple overlapping boxes for the same plate.
    // NMS removes the duplicates and keeps only the best one.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.1f, 0.9f, indices);
    cout << "Boxes: " << boxes.size() << endl;

    // 6. Draw the final boxes on the original image
    std::vector<PlateResult> results;
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        int classId = classIds[idx];
        float conf = confidences[idx];

        // Draw the rectangle
        cv::rectangle(picture, box, cv::Scalar(0, 255, 128), 2); // Green box, thickness 3

        // Create a label with the class ID and confidence score
        std::string label = "Class " + std::to_string(classId) + " (" + std::to_string(conf).substr(0, 4) + ")";

        // Draw the text
        cv::putText(picture, label, cv::Point(box.x, box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(128, 255, 0), 2);
        PlateResult res;
        res.confidence = conf;
        res.x = box.x;
        res.y = box.y;
        res.w = box.width;
        res.h = box.height;
        results.push_back(res);

    }

    // 7. Display the result!
    cv::imshow("Plate Detection Result", picture);
    //cv::waitKey(0); // Wait until the user presses a key
    return results;
}

#ifndef PLATEFINDER_H
#define PLATEFINDER_H
#include "QString"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "plateresult.h"
#include <vector>
//#include <opencv4/opencv2/opencv.hpp>
#include "opencv2/opencv.hpp"

class PlateFinder
{
private:
    Ort::Env *env;
    Ort::SessionOptions *session_options;
    Ort::Session *session;
public:
    PlateFinder() = delete;
    PlateFinder(QString modelFile);
    std::vector<PlateResult> InspectPicture(cv::Mat &picture, float threshold);
};

#endif // PLATEFINDER_H

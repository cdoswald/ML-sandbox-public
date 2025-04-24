#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

int main (){

    // Specify constants
    const int height = 640;
    const int width = 640;
    const int channels = 3;
    const std::string model_path = "models/yolo11n.onnx";

    // Load YOLOv11 model weights
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv11");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    // Specify input details
    auto input_name = session.GetInputNameAllocated(0, allocator);
    const char* input_names[] = { input_name.get() };
    std::vector<int64_t> input_dims = {1, channels, height, width};
    size_t input_tensor_size = 1 * channels * height * width;

    // Load image
    std::string image_path = "images/cat.jpeg";
    cv::Mat image = cv::imread(image_path);

    // Flatten image
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(height, width));
    resized.convertTo(resized, CV_32F, 1.0 / 255);
    
    std::vector<float> input_tensor_values;
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < resized.rows; ++y) {
            for (int x = 0; x < resized.cols; ++x) {
                input_tensor_values.push_back(resized.at<cv::Vec3f>(y, x)[c]);
            }
        }
    }

    // Create input tensor object
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_size,
        input_dims.data(),
        input_dims.size()
    );

    // Forward pass
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* output_names[] = { output_name.get() };
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    // Extract output
    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    size_t output_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<int64_t> output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

    std::cout << "Output tensor size: " << output_size << std::endl;
    std::cout << "Output shape: ";
    for (const auto& dim : output_shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // Process inference result
    int dimensions = output_shape.at(1);
    int num_detections = output_shape.at(2);

    float conf_threshold = 0.5;

    for (int i = 0; i < num_detections; ++i) {

        // Get object probability
        float obj_conf = output_data[i * dimensions + 4];
        if (obj_conf < conf_threshold) continue;

        // Get highest score class
        float max_score = 0;
        int class_id = -1;
        for (int c = 5; c < dimensions; ++c) {
            float score = output_data[i * dimensions + c];
            if (score > max_score) {
                max_score = score;
                class_id = c - 5;
            }
        }

        if (max_score * obj_conf < conf_threshold) continue;

        // Get bounding box predictions
        float cx = output_data[i * dimensions + 0];
        float cy = output_data[i * dimensions + 1];
        float w = output_data[i * dimensions + 2];
        float h = output_data[i * dimensions + 3];

        std::cout << "Bbox predictions (not scaled): (" << cx << ", " << cy << ", " << w << ", " << h << ")" << std::endl;

        // Convert bounding box predictions from center (x,y) to top-left (x,y)
        int x = static_cast<int>((cx - w/2.0) * resized.cols);
        int y = static_cast<int>((cy - h/2.0) * resized.rows);
        int bbox_width = static_cast<int>(w * resized.cols);
        int bbox_height = static_cast<int>(h * resized.rows);

        std::cout << "Bounding Box: (" << x << ", " << y << ", " << bbox_width << ", " << bbox_height << ")" << std::endl;

        cv::rectangle(resized, cv::Rect(cx, cy, w, h), cv::Scalar(0, 255, 0), 2);
        cv::putText(
            resized, std::to_string(class_id), cv::Point(cx, cy-10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1
        );

    }

    // Display image with bounding box
    cv::imshow("YOLOv11 Output:", resized);
    cv::waitKey(0);
    return 0;
}

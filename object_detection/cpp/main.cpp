// Real-time object detection with YOLOv11.

#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using json = nlohmann::json;


int check_video_rotation(const std::string &filepath) {
    // Use ffmpeg package to check .mp4 file metadata
    std::string cmd = "ffprobe -show_entries side_data -print_format json \"" + filepath + "\"";
    std::string result;
    char buffer[128];
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return -1;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    // Extract rotation degrees from metadata string
    std::regex rot_regex_pattern{R"("rotation":\s*(-?\d+))"}; // inner parentheses indicate 1st capture group
    std::smatch match;
    if (std::regex_search(result, match, rot_regex_pattern)) {
        return std::stoi(match[1].str());
    }
    // If no rotation, return 0 degrees
    return 0;
}

int main (){

    // Read config file
    std::ifstream f("config.json");
    json config = json::parse(f);

    // Extract constants from config
    const std::string input_path = config.at("input_path");
    const std::string output_path = config.at("output_path");
    const std::string model_path = config.at("YOLO_model_path");
    const bool real_time_display = config.at("real_time_display");

    // Define rotations to apply based on video metadata
    // (note that this differs from Python version)
    std::map<int, int> frame_rot_codes = { 
        {-90, cv::ROTATE_180},
        {90, cv::ROTATE_180}
    };

    // Check if original video is rotated (e.g., iPhone camera)
    const int frame_rot_deg = check_video_rotation(input_path);
    if (frame_rot_deg == -1) throw std::runtime_error("Video metadata check was unsuccessful.");
    bool rotate = frame_rot_codes.count(frame_rot_deg) > 0;
    int rotation_code = rotate? frame_rot_codes[frame_rot_deg] : -1;

    // Create ONNX runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv11");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    // Get expected model input dims
    auto input_name = session.GetInputNameAllocated(0, allocator);
    const char* input_names[] = { input_name.get() };
    size_t num_inputs = session.GetInputCount();
    if (num_inputs != 1) {
        std::cerr << "Error: Model was expected to have 1 input, but " 
            << num_inputs << " inputs reported." << std::endl;
        throw std::runtime_error("Unexpected number of model inputs");
    }
    auto input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();

    const int input_channels = input_dims.at(1);
    const int input_height = input_dims.at(2);
    const int input_width = input_dims.at(3);

    // Start video capture
    cv::VideoCapture cap{input_path};
    cv::Mat frame;

    // Determine delay time (for breaking early)
    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = static_cast<int>(1000 / fps);

    // Start object detection
    while (cap.isOpened()) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Preprocess image
        if (rotate) {
            cv::rotate(frame, frame, rotation_code);
        }
        cv::Mat proc_frame = cv::dnn::blobFromImage(
            frame,
            1.0/255.0, // rescale
            cv::Size(input_width, input_height), // resize
            {}, // don't subtract mean
            true //swap color channels (BGR -> RGB)
        );
    
        // Create input tensor object
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            proc_frame.ptr<float>(),
            proc_frame.total(),
            input_dims.data(),
            input_dims.size()
        );

        // Run model forward pass
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        const char* output_names[] = { output_name.get() };
        auto output = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            num_inputs,
            output_names,
            1 // output count
        );

        // Extract output
        std::vector<int64_t> output_shape = output.front().GetTensorTypeAndShapeInfo().GetShape();
        size_t output_size = output.front().GetTensorTypeAndShapeInfo().GetElementCount();
        float* output_data = output.front().GetTensorMutableData<float>();

        // std::cout << "Output tensor size: " << output_size << std::endl;
        // std::cout << "Output shape: ";
        // for (const auto &dim : output_shape) {
        //     std::cout << dim << " ";
        // }
        // std::cout << std::endl;

        // output is 1 x 84 x 8400 shape (705600 total)

        // Process predicted bounding boxes
        int preds_per_box = output_shape.at(1);
        int num_boxes = output_shape.at(2);

        float conf_threshold = 0.25;
        int num_classes = 80;

        // Loop over all proposed bounding boxes
        std::vector<std::vector<float>> saved_detections;
        for (size_t k = 0; k < num_boxes; ++k) {

            // Get highest probability class for each proposed bbox
            float max_class_prob = 0.0;
            float max_class_idx = -1.0;
            for (size_t m = 0; m < num_classes; ++m) {
                float class_prob = output_data[num_boxes * (4 + m) + k];
                if (class_prob > max_class_prob) {
                    max_class_prob = class_prob;
                    max_class_idx = m;
                }
            }

            // Retain bbox coords and class label
            // if highest class probability is at/above confidence threshold
            if (max_class_prob >= conf_threshold) {
                float x_center = output_data[0 * num_boxes + k];
                float y_center = output_data[1 * num_boxes + k];
                float bbox_width = output_data[2 * num_boxes + k];
                float bbox_height = output_data[3 * num_boxes + k];

                // Convert to (x1, y1), (x2, y2) coords
                float x1 = x_center - bbox_width / 2;
                float y1 = y_center - bbox_height / 2;
                float x2 = x_center + bbox_width / 2;
                float y2 = y_center + bbox_height / 2;
                saved_detections.push_back(
                    {x1, y1, x2, y2, max_class_idx, max_class_prob}
                );
            }
        }

        // Loop over final bounding boxes and display on image
        for (std::vector<float> bbox_data : saved_detections) {

            // Scale bbox coords to match original frame dimensions
            float scale_x = frame.cols / static_cast<float>(input_width);
            float scale_y = frame.rows / static_cast<float>(input_height);

            int x1 = static_cast<int>(bbox_data.at(0) * scale_x);
            int y1 = static_cast<int>(bbox_data.at(1) * scale_y);
            int x2 = static_cast<int>(bbox_data.at(2) * scale_x);
            int y2 = static_cast<int>(bbox_data.at(3) * scale_y);

            cv::rectangle(
                frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2
            );
            cv::putText(
                frame, std::to_string(bbox_data.at(4)), cv::Point(x1, y1),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1
            );
        }

        cv::imshow("Video", frame);
        if (cv::waitKey(delay) >= 0) {
            break;
        }

    }

    // Clean-up
    cap.release();
    cv::destroyAllWindows();

    return 0;
}

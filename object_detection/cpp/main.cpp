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







        cv::imshow("Video", frame);
        if (cv::waitKey(delay) >= 0) {
            break;
        }

    }

    // Clean-up
    cap.release();
    cv::destroyAllWindows();

    return 0;






    // // Extract output
    // float* output_data = output_tensors.front().GetTensorMutableData<float>();
    // size_t output_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    // std::vector<int64_t> output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

    // std::cout << "Output tensor size: " << output_size << std::endl;
    // std::cout << "Output shape: ";
    // for (const auto& dim : output_shape) {
    //     std::cout << dim << " ";
    // }
    // std::cout << std::endl;

    // // Process inference result
    // int dimensions = output_shape.at(1);
    // int num_detections = output_shape.at(2);

    // float conf_threshold = 0.5;

    // for (int i = 0; i < num_detections; ++i) {

    //     // Get object probability
    //     float obj_conf = output_data[i * dimensions + 4];
    //     if (obj_conf < conf_threshold) continue;

    //     // Get highest score class
    //     float max_score = 0;
    //     int class_id = -1;
    //     for (int c = 5; c < dimensions; ++c) {
    //         float score = output_data[i * dimensions + c];
    //         if (score > max_score) {
    //             max_score = score;
    //             class_id = c - 5;
    //         }
    //     }

    //     if (max_score * obj_conf < conf_threshold) continue;

    //     // Get bounding box predictions
    //     float cx = output_data[i * dimensions + 0];
    //     float cy = output_data[i * dimensions + 1];
    //     float w = output_data[i * dimensions + 2];
    //     float h = output_data[i * dimensions + 3];

    //     std::cout << "Bbox predictions (not scaled): (" << cx << ", " << cy << ", " << w << ", " << h << ")" << std::endl;

    //     // Convert bounding box predictions from center (x,y) to top-left (x,y)
    //     int x = static_cast<int>((cx - w/2.0) * resized.cols);
    //     int y = static_cast<int>((cy - h/2.0) * resized.rows);
    //     int bbox_width = static_cast<int>(w * resized.cols);
    //     int bbox_height = static_cast<int>(h * resized.rows);

    //     std::cout << "Bounding Box: (" << x << ", " << y << ", " << bbox_width << ", " << bbox_height << ")" << std::endl;

    //     cv::rectangle(resized, cv::Rect(cx, cy, w, h), cv::Scalar(0, 255, 0), 2);
    //     cv::putText(
    //         resized, std::to_string(class_id), cv::Point(cx, cy-10),
    //         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1
    //     );

    // }

    // // Display image with bounding box
    // cv::imshow("YOLOv11 Output:", resized);
    // cv::waitKey(0);
    // return 0;
}

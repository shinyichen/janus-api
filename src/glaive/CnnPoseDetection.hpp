#ifndef CNNPOSE_HPP
#define CNNPOSE_HPP

#include <unordered_map>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include "caffe/caffe.hpp"

#include "glaive_common.hpp"
#include "iarpa_janus.h"

class CnnPoseDetection
{
public:
  CnnPoseDetection() {};

  janus_error initialize(std::string sdk_path, const int gpu_dev);
  int detect_pose(cv::Mat img_in, int face_x, int face_y, int face_w, int face_h);

private:
  caffe::Net<float> *pose_net;
  int cnn_feature_dimension_size;
  std::string m_feature_layer;
  int target_width, target_height;
};



#endif

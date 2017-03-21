#ifndef CNNFEATEX_HPP
#define CNNFEATEX_HPP

#include <unordered_map>
#include <vector>
#include <string>
#include <atomic>
#include <opencv2/core/core.hpp>
#include "caffe/caffe.hpp"

#include "glaive_common.hpp"
#include "iarpa_janus.h"

class CnnFeatex
{
public:
  CnnFeatex() { m_initialized.store(false); };

  janus_error initialize(std::string sdk_path, const int gpu_dev);
  std::vector<featv_t> extract_batch(std::vector<cv::Mat*>);

private:
  caffe::Net<float> *feature_extraction_net;
  int cnn_feature_dimension_size;
  caffe::Blob<float> mean_image_blob;
  std::string m_feature_layer;
  std::atomic<bool> m_initialized;

  bool CvImageListToBlob(std::vector<cv::Mat*> image_list, int target_height, int target_width, const caffe::Blob<float> &mean_image_blob, caffe::Blob<float>* out_blob);
};



#endif

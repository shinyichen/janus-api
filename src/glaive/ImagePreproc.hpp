#ifndef IMAGEPREPROC_HPP
#define IMAGEPREPROC_HPP

#include "glaive_common.hpp"
#include "iarpa_janus.h"

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>

class ImagePreproc
{
public:
  ImagePreproc() {};
  janus_error initialize(std::string sdk_path);
  janus_error convertJanusMediaImageToOpenCV(janus_media media, int frame_num, cv::Mat &image_out);
  janus_error process(cv::Mat image_in, janus_attributes metadata_attributes, cv::Mat &image_out);
  janus_error process_withrender(cv::Mat image_in, janus_attributes metadata_attributes, int image_type, cv::Mat &out_cropped, cv::Mat &out_rend_fr, cv::Mat &out_rend_hp, cv::Mat &out_rend_fp, cv::Mat &out_aligned, float &out_yaw);

private:
  float m_landmark_conf_threshold;

  // Big-picture functions
  janus_error detect_landmarks(cv::Mat image_in, janus_attributes metadata_attributes,
			       std::vector<cv::Point2f> &out_landmarks, float &out_landmark_confidence);
  janus_error detect_landmarks_noanchors(cv::Mat image_in, janus_attributes metadata_attributes, int image_type,
					 std::vector<cv::Point2f> &out_landmarks, float &out_landmark_confidence);
  janus_error align_frontal(cv::Mat image_in, std::vector<cv::Point2f> landmarks_in, cv::Mat &image_out);
  janus_error align_profile(cv::Mat image_in, std::vector<cv::Point2f> landmarks_in, float yaw, cv::Mat &image_out);


  // Helper functions
  Eigen::Matrix<float, 3, 3> findNonReflexiveSimilarityTransform(std::vector<cv::Point2f> source, std::vector<cv::Point2f> target);
  std::vector<cv::Point2f> transformPoints(Eigen::Matrix<float, 3, 3> transformation,  std::vector<cv::Point2f> points);
  float evaluateTransformation(std::vector<cv::Point2f> candidate, std::vector<cv::Point2f> target);
  Eigen::Matrix<float, 3, 3> findSimilarityTransform(std::vector<cv::Point2f> source, std::vector<cv::Point2f> target);
  cv::Mat convertEigenToMat(Eigen::MatrixXf eigen);
};



#endif

#ifndef PYTHON_WRAPPERS_HPP
#define PYTHON_WRAPPERS_HPP

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

#include "iarpa_janus.h"

janus_error pythonInitialize(std::string pythonPath, std::string poseConfigFile);

janus_error pythonGetPose(size_t imageHeight, size_t imageWidth, int stride, int cv_type, uint8_t *imageData,
			  const std::vector<cv::Point2f> lm, int normalizationScale,
			  cv::Mat &image_n, std::vector<cv::Point2f> &pose_landmarks );

janus_error pythonDoRendering(cv::Mat image_in, std::vector<cv::Point2f> lmrks_in, cv::Mat &rend_fr_out, cv::Mat &rend_hp_out, cv::Mat &rend_fp_out, float &yaw_out);

janus_error pythonFinalize();

#endif

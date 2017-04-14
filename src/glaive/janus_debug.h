#ifndef JANUS_DEBUG_H
#define JANUS_DEBUG_H

#include "iarpa_janus.h"
#include <vector>
#include <opencv2/core/core.hpp>

JANUS_EXPORT janus_error janus_debug(janus_association &association,
												const janus_template_role role,
												cv::Mat &out_cropped,
												cv::Mat &out_rend_fr,
												cv::Mat &out_rend_hp,
												cv::Mat &out_rend_fp,
												cv::Mat &out_aligned,
												float &out_yaw,
												std::vector<cv::Point2f> &out_landmarks,
												float &out_confidence);

// JANUS_EXPORT janus_error janus_create_template_debug(
// 	std::vector<janus_association> &associations,
// 	const janus_template_role role,
// 	janus_template &template_,
// 	std::vector<cv::Mat> &out_cropped,
// 	std::vector<cv::Mat> &out_rend_fr,
// 	std::vector<cv::Mat> &out_rend_hp,
// 	std::vector<cv::Mat> &out_rend_fp,
// 	std::vector<cv::Mat> &out_aligned,
	// float out_yaw[],
	// std::vector<std::vector<cv::Point2f>> &out_landmarks,
	// float out_confidence[]);

#endif

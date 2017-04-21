#include "ImagePreproc.hpp"
#include "CLM.h"
#include "CLM_utils.h"
#include "JanusUtils.h"
#include "python_wrappers.h"

#include <libconfig.h++>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

janus_error ImagePreproc::initialize(std::string sdk_path)
{
  libconfig::Config cfg;
  std::string preproc_config_file = sdk_path + "/preproc.config";

  try {
    cfg.readFile(preproc_config_file.c_str());
  } catch(const libconfig::FileIOException &fioex) {
    std::cerr << "Could not open " << preproc_config_file << " for reading." << std::endl;
    return JANUS_OPEN_ERROR;
  } catch (const libconfig::ParseException &pex) {
    std::cerr << "Could not parse " << preproc_config_file << "." << std::endl;
    return JANUS_PARSE_ERROR;
  }

  std::string landmark_model;

  try {
    cfg.lookupValue("landmark_model", landmark_model);
    cfg.lookupValue("landmark_confidence_threshold", m_landmark_conf_threshold);
  } catch (const libconfig::SettingNotFoundException &nfex) {
    std::cerr << "Could not find setting in preproc.config file." << std::endl;
    return JANUS_UNKNOWN_ERROR;
  }

  CLMJanus::initialize_lm_detector(sdk_path + "/" + landmark_model);

  return JANUS_SUCCESS;
}

janus_error ImagePreproc::convertJanusMediaImageToOpenCV(janus_media media, int frame_num, cv::Mat &image_out)
{
  // First determine color-space of image, and convert to OpenCV color-space type
  int cv_type;
  if      (media.color_space == JANUS_GRAY8)   {cv_type = CV_8UC1;}
  else if (media.color_space == JANUS_BGR24)   {cv_type = CV_8UC3;}
  else                 			       { return JANUS_INVALID_MEDIA;}

  // Now create cv::Mat wrapper around data pointer, and set output var
  cv::Mat cv_img(media.height, media.width, cv_type, media.data[frame_num], media.step);
  image_out = cv_img;

  return JANUS_SUCCESS;
}

janus_error ImagePreproc::process(cv::Mat image_in, janus_attributes metadata_attributes, cv::Mat &image_out)
{
  janus_error status;

  std::vector<cv::Point2f> landmarks;
  float landmark_confidence;

  int image_type = 0; // XX NEED TO DO THIS
  status = detect_landmarks_noanchors(image_in, metadata_attributes, image_type, landmarks, landmark_confidence);
  //status = detect_landmarks(image_in, metadata_attributes, landmarks, landmark_confidence);
  if (status != JANUS_SUCCESS) return status;

  std::cout << "About to do align image..." << std::endl;
  status = align_frontal(image_in, landmarks, image_out);
  std::cout << "Done with align image." << std::endl;
  if (status != JANUS_SUCCESS) return status;


  return JANUS_SUCCESS;
}

janus_error do_crop(cv::Mat image_in, janus_attributes metadata_attributes, cv::Mat &image_out) {

  std::cout << "Insided do_crop()" << std::endl;
  float rescaleFrontal[] = {1.4421, 2.2853, 1.4421, 1.4286};
  float rescaleProfile[] = {0.8138, 1.8524, 2.1056, 1.3463};
  int frontalSize[] = {159,122};
  int profileSize[] = {230,210};
  float rescaleCS2[] = {1.7342,2.3882,1.7342,1.0801};

  //metadata_attributes.face_x;
  float cx = metadata_attributes.face_x + 0.5 * metadata_attributes.face_width;
  float cy = metadata_attributes.face_y + 0.5 * metadata_attributes.face_height;
  float tsize = metadata_attributes.face_width/2.0;
  if (metadata_attributes.face_height > metadata_attributes.face_width)
    tsize = metadata_attributes.face_height/2.0;

  float l = cx - tsize;
  float t = cy - tsize;
  cx = l + (2*tsize)/(rescaleCS2[0]+rescaleCS2[2]) * rescaleCS2[0];
  cy = t + (2*tsize)/(rescaleCS2[1]+rescaleCS2[3]) * rescaleCS2[1];
  tsize = 2*tsize/(rescaleCS2[0]+rescaleCS2[2]);

  int bl = int(cx - rescaleFrontal[0]*tsize);
  int bt = int(cy - rescaleFrontal[1]*tsize);
  int br = int(cx + rescaleFrontal[2]*tsize);
  int bb = int(cy + rescaleFrontal[3]*tsize);
  int nw = br-bl;
  int nh = bb-bt;

  int ll = 0;
  if (bl < 0) {
    ll = -bl;
    bl = 0;
  }

  int rr = nw;
  if (br > image_in.cols) {
    rr = image_in.cols+nw - br;
    br = image_in.cols;
  }

  int tt = 0;
  if (bt < 0) {
    tt = -bt;
    bt = 0;
  }

  int bbb = nh;
  if (bb > image_in.rows) {
    bbb = image_in.rows+nh - bb;
    bb = image_in.rows;
  }

  // Look into speeding this up (vectorize it? implement via OpenCV?)

  if (nh == 0 || nw == 0) {
    std::cout << "bad face bounding box, outside geometry of image" << std::endl;
    return JANUS_UNKNOWN_ERROR;
  }

  image_out = cv::Mat::zeros(nh, nw, CV_8UC3);

  // Sanity check
  if (bbb-tt != bb-bt || rr-ll != br-bl) {
    std::cout << "Failed Sanity Check in crop() function" << std::endl;
    return JANUS_UNKNOWN_ERROR;
  }

  for (int row_dest=tt, row_source=bt; row_dest < bbb; ++row_dest, ++row_source) {
    for (int col_dest=ll, col_source=bl; col_dest < rr; ++col_dest, ++col_source) {
      image_out.at<cv::Vec3b>(row_dest, col_dest) = image_in.at<cv::Vec3b>(row_source, col_source);
    }
  }

  std::cout << "Done with do_crop()" << std::endl;

  return JANUS_SUCCESS;
}

janus_error ImagePreproc::process_withrender_debug(cv::Mat image_in, janus_attributes metadata_attributes, int image_type,
                                                   cv::Mat &out_cropped, cv::Mat &out_rend_fr, cv::Mat &out_rend_hp, cv::Mat &out_rend_fp, cv::Mat &out_aligned, float &out_yaw, std::vector<cv::Point2f> &landmarks, float &landmark_confidence,
                                                   unsigned int &landmark_dur, unsigned int &render_dur, unsigned int &align_dur)
{
  janus_error status;

  // std::vector<cv::Point2f> landmarks;
  // float landmark_confidence;
  auto start_time = std::chrono::steady_clock::now();

  // Run Landmark Detection
  if (image_type != 0) {
    std::cout << "About to run LM Detection" << std::endl;
    status = detect_landmarks_noanchors(image_in, metadata_attributes, image_type, landmarks, landmark_confidence);
    std::cout << "\tlm confidence: " << landmark_confidence << "\t(Threshold confidence = " << m_landmark_conf_threshold << ")" << std::endl;
  } else {
    status = JANUS_UNKNOWN_ERROR;
  }

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
  landmark_dur = duration.count();

  if (status != JANUS_SUCCESS || landmark_confidence < m_landmark_conf_threshold) {
    // Now Do Cropping
    std::cout << "Doing Cropped step" << std::endl;
    status = do_crop(image_in, metadata_attributes, out_cropped);
    std::cout << "Done Cropped step: status = " << status << std::endl;

    if (status != JANUS_SUCCESS) return status;

    // Early Return -- Lm failed so only can do cropped image
    return JANUS_SUCCESS;
  }

  // After running landmark, need to run lm2pose
  cv::Mat image_poseCorrected;
  std::vector<cv::Point2f> landmarks_poseCorrected;

  std::cout << "About to run lm2pose" << std::endl;
  cv::Mat image_in_copy = image_in.clone();
  status = pythonGetPose(image_in_copy.rows, image_in_copy.cols, image_in_copy.channels(), image_in_copy.type(),
                         image_in_copy.data, landmarks, 1, image_poseCorrected, landmarks_poseCorrected);
  if (status != JANUS_SUCCESS) {
    // getPose failed, need to do a graceful fallback
    // can't do rendering w/o pose, but let's do align-frontal and put it in the 'cropped' image so
    // create-template code doesn't try to do pooling
    status = align_frontal(image_in, landmarks, out_cropped);

    return status;
  }

  // Now do render
  std::cout << "About to do rendering" << std::endl;
  start_time = std::chrono::steady_clock::now();
  status = pythonDoRendering(image_poseCorrected, landmarks_poseCorrected, out_rend_fr, out_rend_hp, out_rend_fp, out_yaw);
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
  render_dur = duration.count();

  if (status != JANUS_SUCCESS) return status;

  std::cout << "About to do align image..." << std::endl;

  start_time = std::chrono::steady_clock::now();
  if (fabs(out_yaw) < 30) {
    status = align_frontal(image_in, landmarks, out_aligned);
  } else {
    status = align_profile(image_in, landmarks, out_yaw, out_aligned);
  }
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
  align_dur = duration.count();

  if (status != JANUS_SUCCESS) return status;

  std::cout << "Done with align image." << std::endl;

  return JANUS_SUCCESS;
}

janus_error ImagePreproc::process_withrender(cv::Mat image_in, janus_attributes metadata_attributes, int image_type, cv::Mat &out_cropped, cv::Mat &out_rend_fr, cv::Mat &out_rend_hp, cv::Mat &out_rend_fp, cv::Mat &out_aligned, float &out_yaw)
{
  janus_error status;

  std::vector<cv::Point2f> landmarks;
  float landmark_confidence;

  // Run Landmark Detection
  if (image_type != 0) {
    std::cout << "About to run LM Detection" << std::endl;
    status = detect_landmarks_noanchors(image_in, metadata_attributes, image_type, landmarks, landmark_confidence);
    std::cout << "\tlm confidence: " << landmark_confidence << "\t(Threshold confidence = " << m_landmark_conf_threshold << ")" << std::endl;
  } else {
    status = JANUS_UNKNOWN_ERROR;
  }

  if (status != JANUS_SUCCESS || landmark_confidence < m_landmark_conf_threshold) {
    // Now Do Cropping
    std::cout << "Doing Cropped step" << std::endl;
    status = do_crop(image_in, metadata_attributes, out_cropped);
    std::cout << "Done Cropped step: status = " << status << std::endl;

    if (status != JANUS_SUCCESS) return status;

    // Early Return -- Lm failed so only can do cropped image
    return JANUS_SUCCESS;
  }

  // After running landmark, need to run lm2pose
  cv::Mat image_poseCorrected;
  std::vector<cv::Point2f> landmarks_poseCorrected;

  std::cout << "About to run lm2pose" << std::endl;
  cv::Mat image_in_copy = image_in.clone();
  status = pythonGetPose(image_in_copy.rows, image_in_copy.cols, image_in_copy.channels(), image_in_copy.type(),
                         image_in_copy.data, landmarks, 1, image_poseCorrected, landmarks_poseCorrected);

  if (status != JANUS_SUCCESS) {
    // getPose failed, need to do a graceful fallback
    // can't do rendering w/o pose, but let's do align-frontal and put it in the 'cropped' image so
    // create-template code doesn't try to do pooling
    status = align_frontal(image_in, landmarks, out_cropped);

    return status;
  }

  // Now do render
  std::cout << "About to do rendering" << std::endl;
  status = pythonDoRendering(image_poseCorrected, landmarks_poseCorrected, out_rend_fr, out_rend_hp, out_rend_fp, out_yaw);

  if (status != JANUS_SUCCESS) return status;

  std::cout << "About to do align image..." << std::endl;

  if (fabs(out_yaw) < 30) {
    status = align_frontal(image_in, landmarks, out_aligned);
  } else {
    status = align_profile(image_in, landmarks, out_yaw, out_aligned);
  }

  if (status != JANUS_SUCCESS) return status;

  std::cout << "Done with align image." << std::endl;

  return JANUS_SUCCESS;
}

janus_error ImagePreproc::detect_landmarks(cv::Mat image_in, janus_attributes metadata_attributes, std::vector<cv::Point2f> &out_landmarks, float &out_landmark_confidence)
{
  std::vector<float> seed_landmarks(10);
  seed_landmarks[0] = metadata_attributes.face_x;
  seed_landmarks[1] = metadata_attributes.face_y;
  seed_landmarks[2] = metadata_attributes.face_width;
  seed_landmarks[3] = metadata_attributes.face_height;
  seed_landmarks[4] = metadata_attributes.right_eye_x;
  seed_landmarks[5] = metadata_attributes.right_eye_y;
  seed_landmarks[6] = metadata_attributes.left_eye_x;
  seed_landmarks[7] = metadata_attributes.left_eye_y;
  seed_landmarks[8] = metadata_attributes.nose_base_x;
  seed_landmarks[9] = metadata_attributes.nose_base_y;


  for (size_t i = 0; i < 10; ++i) {
    if (seed_landmarks[i] != seed_landmarks[i]) {
      seed_landmarks[i] = 0;
      std::cout << "Fixing NaN to 0" << std::endl;
    }
  }

  std::cout << "About to do landmark detection: Seed landmarks =" << std::endl;
  std::cout << "\t Face X,Y = (" << seed_landmarks[0] << "," << seed_landmarks[1] << ")" << std::endl;
  std::cout << "\t Face Width,Height = (" << seed_landmarks[2] << "," << seed_landmarks[3] << ")" << std::endl;
  std::cout << "\t Right Eye X,Y = (" << seed_landmarks[4] << "," << seed_landmarks[5] << ")" << std::endl;
  std::cout << "\t Left Eye X,Y = (" << seed_landmarks[6] << "," << seed_landmarks[7] << ")" << std::endl;
  std::cout << "\t Face Nose Base X,Y = (" << seed_landmarks[8] << "," << seed_landmarks[9] << ")" << std::endl;


  if (CLMJanus::detect_landmarks(image_in, seed_landmarks, &out_landmarks, &out_landmark_confidence) != 0)
    return JANUS_UNKNOWN_ERROR;

  std::cout << "Done landmark detection!" << std::endl;

  return JANUS_SUCCESS;
}


janus_error ImagePreproc::detect_landmarks_noanchors(cv::Mat image_in, janus_attributes metadata_attributes, int image_type, std::vector<cv::Point2f> &out_landmarks, float &out_landmark_confidence)
{
  std::vector<float> seed_landmarks(4);
  seed_landmarks[0] = metadata_attributes.face_x;
  seed_landmarks[1] = metadata_attributes.face_y;
  seed_landmarks[2] = metadata_attributes.face_width;
  seed_landmarks[3] = metadata_attributes.face_height;

  std::cout << "About to do landmark detection: w/ no Seed landmarks" << std::endl;
  std::cout << "\t Face X,Y = (" << seed_landmarks[0] << "," << seed_landmarks[1] << ")" << std::endl;
  std::cout << "\t Face Width,Height = (" << seed_landmarks[2] << "," << seed_landmarks[3] << ")" << std::endl;

  cv::Mat image_in_copy = image_in.clone();
  if (CLMJanus::detect_landmarks_noanchors(image_in_copy, seed_landmarks, image_type, &out_landmarks, &out_landmark_confidence) != 0) {
    std::cout << "Landmark detection failed! Falling back to cropping" << std::endl;
    return JANUS_UNKNOWN_ERROR;
  }

  std::cout << "Done landmark detection!" << std::endl;

  return JANUS_SUCCESS;
}

janus_error ImagePreproc::align_profile(cv::Mat image_in, std::vector<cv::Point2f> landmarks_in, float yaw, cv::Mat &image_out)
{
  vector<cv::Point2f> profile_reference_landmarks;
  profile_reference_landmarks.push_back( cv::Point2f(71.0, 100.0) );
  profile_reference_landmarks.push_back( cv::Point2f(58.0, 146) );

  std::vector<cv::Point2f> source_landmarks;
  source_landmarks.push_back( landmarks_in[36] );
  source_landmarks.push_back( landmarks_in[39] );
  source_landmarks.push_back( landmarks_in[42] );
  source_landmarks.push_back( landmarks_in[45] );
  source_landmarks.push_back( landmarks_in[50] );
  source_landmarks.push_back( landmarks_in[51] );
  source_landmarks.push_back( landmarks_in[52] );
  source_landmarks.push_back( landmarks_in[48] );
  source_landmarks.push_back( landmarks_in[54] );

  cv::Mat image_flipped;
  std::vector<Point2f> landmarks_flipped(landmarks_in);
  if (yaw > 0) {
    // flip image and landmarks
    std::cout << "alignProfile: flipping " << endl;

    cv::flip(image_in, image_flipped, 1);
    for (size_t i=0; i<landmarks_in.size(); ++i) {
      landmarks_flipped[i].x = image_in.cols - landmarks_in[i].x;
    }

  } else {
    image_flipped = image_in;
  }

  float eyesLeftX = landmarks_flipped[36].x + landmarks_flipped[37].x + landmarks_flipped[38].x + landmarks_flipped[39].x + landmarks_flipped[40].x + landmarks_flipped[41].x;
  eyesLeftX = eyesLeftX / 6.f;

  float eyesLeftY = landmarks_flipped[36].y + landmarks_flipped[37].y + landmarks_flipped[38].y + landmarks_flipped[39].y + landmarks_flipped[40].y + landmarks_flipped[41].y;
  eyesLeftY = eyesLeftY / 6.f;

  float eyesRightX = landmarks_flipped[42].x + landmarks_flipped[43].x + landmarks_flipped[44].x + landmarks_flipped[45].x + landmarks_flipped[46].x + landmarks_flipped[47].x;
  eyesRightX = eyesRightX / 6.f;

  float eyesRightY = landmarks_flipped[42].y + landmarks_flipped[43].y + landmarks_flipped[44].y + landmarks_flipped[45].y + landmarks_flipped[46].y + landmarks_flipped[47].y;
  eyesRightY = eyesRightY / 6.f;

  float centerEyesX = (eyesLeftX + eyesRightX) / 2.f;
  float centerEyesY = (eyesLeftY + eyesRightY) / 2.f;

  float noseX = landmarks_flipped[30].x + landmarks_flipped[32].x + landmarks_flipped[33].x + landmarks_flipped[34].x;
  noseX = noseX / 4.f;
  float noseY = landmarks_flipped[30].y + landmarks_flipped[32].y + landmarks_flipped[33].y + landmarks_flipped[34].y;
  noseY = noseY / 4.f;

  std::vector<cv::Point2f> sourcePoints;
  sourcePoints.push_back(cv::Point(centerEyesX, centerEyesY));
  sourcePoints.push_back(cv::Point(noseX, noseY));

  Eigen::Matrix<float, 3, 3> T = findSimilarityTransform(sourcePoints, profile_reference_landmarks);
  T.transposeInPlace();

  cv::Mat transformedImg;
  cv::Mat cvTransform = convertEigenToMat(T);

  cv::warpPerspective(image_flipped, transformedImg, cvTransform, image_flipped.size());

  int rx = 11;
  int ry = 1;
  int rwidth = 208;
  int rheight = 228;


  if (rwidth > transformedImg.cols - rx) {
    rwidth = transformedImg.cols - rx;
  }
  if (rheight > transformedImg.rows - ry) {
    rheight = transformedImg.rows - ry;
  }
  cv::Rect cropped_region = cv::Rect(rx, ry, rwidth, rheight);

  cv::Mat cropped_img(transformedImg, cropped_region);
  image_out = cropped_img;

  return JANUS_SUCCESS;
}

janus_error ImagePreproc::align_frontal(cv::Mat image_in, std::vector<cv::Point2f> landmarks_in, cv::Mat &image_out)
{
  std::vector<cv::Point2f> source_landmarks(9);
  source_landmarks[0] = landmarks_in[36];
  source_landmarks[1] = landmarks_in[39];
  source_landmarks[2] = landmarks_in[42];
  source_landmarks[3] = landmarks_in[45];
  source_landmarks[4] = landmarks_in[50];
  source_landmarks[5] = landmarks_in[51];
  source_landmarks[6] = landmarks_in[52];
  source_landmarks[7] = landmarks_in[48];
  source_landmarks[8] = landmarks_in[54];

  vector<cv::Point2f> frontal_reference_landmarks(9);
  frontal_reference_landmarks[0] = cv::Point2f(25.0347 + 20.0, 34.1580 + 20.0);
  frontal_reference_landmarks[1] = cv::Point2f(34.1802 + 20.0, 34.1659 + 20.0);
  frontal_reference_landmarks[2] = cv::Point2f(44.1943 + 20.0, 34.0936 + 20.0);
  frontal_reference_landmarks[3] = cv::Point2f(53.4623 + 20.0, 33.8063 + 20.0);
  frontal_reference_landmarks[4] = cv::Point2f(34.1208 + 20.0, 45.4179 + 20.0);
  frontal_reference_landmarks[5] = cv::Point2f(39.3564 + 20.0, 47.0043 + 20.0);
  frontal_reference_landmarks[6] = cv::Point2f(44.9156 + 20.0, 45.3628 + 20.0);
  frontal_reference_landmarks[7] = cv::Point2f(31.1454 + 20.0, 53.0275 + 20.0);
  frontal_reference_landmarks[8] = cv::Point2f(47.8747 + 20.0, 52.7999 + 20.0);

  float cx = frontal_reference_landmarks[0].x + frontal_reference_landmarks[1].x +
             frontal_reference_landmarks[2].x + frontal_reference_landmarks[3].x +
             frontal_reference_landmarks[7].x + frontal_reference_landmarks[8].x;
  cx = cx/6.f;

  float top = frontal_reference_landmarks[0].y + frontal_reference_landmarks[1].y + frontal_reference_landmarks[2].y;
  top = top/3.f;

  float bottom = frontal_reference_landmarks[7].y;

  float dx = frontal_reference_landmarks[2].x - frontal_reference_landmarks[0].x;
  float dy = bottom - top;

  float scale = 2.f;

  float x0 = (cx - dx * 1.6) * scale;
  float x1 = (cx + dx * 1.6) * scale;
  float y0 = (top - dy * 2.0) * scale;
  float y1 = (bottom + dy * 1.2) * scale;

  for (size_t i = 0; i < frontal_reference_landmarks.size(); ++i) {
    frontal_reference_landmarks[i].x *= scale;
    frontal_reference_landmarks[i].y *= scale;
  }

  Eigen::Matrix<float, 3, 3> T = findSimilarityTransform(source_landmarks, frontal_reference_landmarks);
  T.transposeInPlace();

  cv::Mat transformedImg;
  cv::Mat cvTransform = convertEigenToMat(T);

  cv::warpPerspective(image_in, transformedImg, cvTransform, image_in.size());

  if (y0 < 0) y0 = 0;
  if (x0 < 0) x0 = 0;
  if (y1 >= transformedImg.rows) y1=transformedImg.rows-1;
  if (x1 >= transformedImg.cols) x1=transformedImg.cols-1;

  cv::Rect cropped_region = cv::Rect(x0,y0,x1-x0,y1-y0 + 1);
  cv::Mat cropped_img(transformedImg, cropped_region);
  image_out = cropped_img;

  return JANUS_SUCCESS;
}


Eigen::Matrix<float, 3, 3> ImagePreproc::findNonReflexiveSimilarityTransform(std::vector<cv::Point2f> source, std::vector<cv::Point2f> target) {
  /*
    From matlab toolbox/images/images/cp2tform.m

    For a nonreflective similarity:

    let sc = s*cos(theta)
    let ss = s*sin(theta)

    [ sc -ss
    [u v] = [x y 1] *   ss  sc
    tx  ty]

    There are 4 unknowns: sc,ss,tx,ty.

    Another way to write this is:

    u = [x y 1 0] * [sc
    ss
    tx
    ty]

    v = [y -x 0 1] * [sc
    ss
    tx
    ty]

    With 2 or more correspondence points we can combine the u equations and
    the v equations for one linear system to solve for sc,ss,tx,ty.

    [ u1  ] = [ x1  y1  1  0 ] * [sc]
    [ u2  ]   [ x2  y2  1  0 ]   [ss]
    [ ... ]   [ ...          ]   [tx]
    [ un  ]   [ xn  yn  1  0 ]   [ty]
    [ v1  ]   [ y1 -x1  0  1 ]
    [ v2  ]   [ y2 -x2  0  1 ]
    [ ... ]   [ ...          ]
    [ vn  ]   [ yn -xn  0  1 ]

    Or rewriting the above matrix equation:
    U = X * r, where r = [sc ss tx ty]'
    so r = X\U.
  */


  // Setup Ax = b

  int N = source.size();

  Eigen::MatrixXf A = Eigen::MatrixXf::Constant(2*N, 4, 0.f);
  Eigen::VectorXf b = Eigen::VectorXf::Constant(2*N, 0.f);

  for (int i = 0; i < N; ++i) {
    A(i, 0) = target[i].x;
    A(i, 1) = target[i].y;
    A(i, 2) = 1;
    A(i, 3) = 0;

    A(N + i, 0) = target[i].y;
    A(N + i, 1) = -target[i].x;
    A(N + i, 2) = 0;
    A(N + i, 3) = 1;

    b(i) = source[i].x;
    b(N+i) = source[i].y;
  }

  Eigen::Vector4f x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
  float sc = x(0);
  float ss = x(1);
  float tx = x(2);
  float ty = x(3);

  Eigen::Matrix<float, 3, 3> Tinv;
  Tinv <<  sc, -ss, 0,
    ss,  sc, 0,
    tx,  ty, 1;

  Eigen::Matrix<float, 3, 3> T = Tinv.inverse();

  T(0,2) = 0;
  T(1,2) = 0;
  T(2,2) = 1;

  return T;
}


std::vector<cv::Point2f> ImagePreproc::transformPoints(Eigen::Matrix<float, 3, 3> transformation,  std::vector<cv::Point2f> points) {
  // First create augmented matrix
  int N = points.size();
  Eigen::MatrixXf augmented_matrix = Eigen::MatrixXf::Constant(N, 3, 0.f);
  for (int i = 0; i < N; ++i) {
    augmented_matrix(i, 0) = points[i].x;
    augmented_matrix(i, 1) = points[i].y;
    augmented_matrix(i, 1) = 1;
  }

  // Now apply the transformation
  Eigen::MatrixXf transformed_matrix = augmented_matrix * transformation;

  // Now generate return value by taking first 2 columns of transformed augmented matrix
  std::vector<cv::Point2f> transformed_points;
  for (int i = 0; i < N; ++i) {
    transformed_points.push_back( cv::Point(transformed_matrix(i,0), transformed_matrix(i,1))  );
  }

  return transformed_points;
}

float ImagePreproc::evaluateTransformation(std::vector<cv::Point2f> candidate, std::vector<cv::Point2f> target) {
  int N = candidate.size();
  float l2norm = 0;
  for (int i = 0; i < N; ++i) {
    float x_diff = target[i].x - candidate[i].x;
    float y_diff = target[i].y - candidate[i].y;
    l2norm += x_diff * x_diff + y_diff*y_diff;
  }
  return l2norm;
}

// From matlab toolbox/images/images/cp2tform.m
// Basically just call findSimilarityTransform on both original points and reflected points
// and pick the one with the smallest L2 norm
//
Eigen::Matrix<float, 3, 3> ImagePreproc::findSimilarityTransform(std::vector<cv::Point2f> source, std::vector<cv::Point2f> target) {
  // Find regular non-reflexive transform
  Eigen::Matrix<float, 3, 3> T1 = findNonReflexiveSimilarityTransform(source, target);

  // Reflect data around y-axis and then find the non-reflexive transform of the reflected data
  int N = source.size();
  std::vector<cv::Point2f> target_reflection;
  for (int i = 0; i < N; i++) {
    target_reflection.push_back( cv::Point2f(- target[i].x, target[i].y) );
  }
  Eigen::Matrix<float, 3, 3> T2 = findNonReflexiveSimilarityTransform(source, target_reflection);

  //  Reflect the transformation to undo original reflection
  Eigen::Matrix<float, 3, 3> undo_reflection;
  undo_reflection << -1, 0, 0,
    0, 1, 0,
    0, 0, 1;

  T2 = T2 * undo_reflection;

  // Now apply both transformations and measure L2 norm of error to see which one is better
  std::vector<cv::Point2f> candidate_1 = transformPoints(T1, source);
  std::vector<cv::Point2f> candidate_2 = transformPoints(T2, source);

  if (evaluateTransformation(candidate_1, target) <= evaluateTransformation(candidate_2, target))
    return T1;
  else
    return T2;

}

cv::Mat ImagePreproc::convertEigenToMat(Eigen::MatrixXf eigen) {
  cv::Mat mat(eigen.rows(), eigen.cols(), CV_32F);

  for (int r = 0; r < eigen.rows(); ++r) {
    for (int c = 0; c < eigen.cols(); ++c) {
      mat.at<float>(r,c) = eigen(r,c);
    }
  }

  return mat;
}

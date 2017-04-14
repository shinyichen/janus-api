#include "iarpa_janus.h"
#include "janus_debug.h"
#include "CnnServer.hpp"
#include "PyServer.hpp"
#include "ImagePreproc.hpp"
#include "yolo.h"

#include <libconfig.h++>
#include <boost/filesystem.hpp>
#include <thread>
#include "cblas.h"
#include <limits>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <mlpack/core.hpp>

#include "python_wrappers.h"

#include "CnnPoseDetection.hpp"

// For now keep these here: later merge into python wrappers
#include <Python.h>
#include <numpy/arrayobject.h>
#include <opencv2/imgproc.hpp>

struct janus_template_type {
  featv_t pooled_feat;
};

struct janus_gallery_type {
  std::vector<size_t> template_ids;
  size_t feat_dim;

  float *feats;
};


CnnServer cnnServer;
PyServer  pyServer;
ImagePreproc preproc;
float face_detect_probabiltiy_threshold, face_detect_nms_threshold;
cv::Mat backupPCATransform, backupPCAMean;
bool g_is_python_up = false;
CnnPoseDetection pose_detector;
arma::mat  galleryData;

// for now keep these here; later merge into python wrappers
std::string py_env_dir, cluster_module, cluster_func, tracker_module, tracker_func;
bool py_init = false;

janus_error load_pca(std::string pca_file, cv::Mat &out_PCATransform, cv::Mat &out_PCAMean);

JANUS_EXPORT janus_error janus_initialize(const std::string &sdk_path,
                                          const std::string &temp_path,
                                          const std::string &algorithm,
                                          const int gpu_dev)
{
  if (algorithm != "noinit" && algorithm != "detectonly") {

    /////////////////////////////////////
    // Do python
    // Initialize Python interpreter + load needed python scripts
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

      std::string pythonPath, pose_config;

      try {
        cfg.lookupValue("pythonPath", pythonPath);
        cfg.lookupValue("pose_config", pose_config);

        pose_config = sdk_path + "/" + pose_config;

        pythonPath = pythonPath + ":" + sdk_path + "/models/pose";
        pythonPath = pythonPath + ":" + sdk_path + "/models/render";

      } catch (const libconfig::SettingNotFoundException &nfex) {
        std::cerr << "Could not find pose python script config info." << std::endl;
        return JANUS_UNKNOWN_ERROR;
      }

      std::cout << "About to initialize python with: pythonPath = " << pythonPath << std::endl;
      std::cout << "About to initialize python with: poseConfig = " << pose_config << std::endl;

      janus_error status = pythonInitialize(pythonPath, pose_config);
      g_is_python_up = true;
      if (status != JANUS_SUCCESS) {
        std::cerr << "Error: Could not initialize python!" << std::endl;
        return status;
      }

    }
    // Done Python
    /////////////////////////////////////

    // Start up CNN server (only one globally, across all processes on a machine)
    janus_error status = cnnServer.spawnCnnWorker(sdk_path, gpu_dev);
    if (status != JANUS_SUCCESS) return status;

    // Initialize Preproc Class (one per process)
    status = preproc.initialize(sdk_path);
    if (status != JANUS_SUCCESS) return status;
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Load backup PCA model
  {
    libconfig::Config cfg;
    std::string pca_config_file = sdk_path + "/pca.config";
    try {
      cfg.readFile(pca_config_file.c_str());
    } catch(const libconfig::FileIOException &fioex) {
      std::cerr << "Could not open " << pca_config_file << " for reading." << std::endl;
      return JANUS_OPEN_ERROR;
    } catch (const libconfig::ParseException &pex) {
      std::cerr << "Could not parse " << pca_config_file << "." << std::endl;
      return JANUS_PARSE_ERROR;
    }

    std::string backup_pca_file;

    try {
      cfg.lookupValue("backup_pca", backup_pca_file);
      backup_pca_file = sdk_path + "/" + backup_pca_file;
    } catch (const libconfig::SettingNotFoundException &nfex) {
      std::cerr << "Could not find setting in pca.config file." << std::endl;
      return JANUS_UNKNOWN_ERROR;
    }

    janus_error status = load_pca(backup_pca_file, backupPCATransform, backupPCAMean);
    if (status != JANUS_SUCCESS) return status;
  }
  // Done with backup PCA model
  ////////////////////////////////////////////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Load python modules and start python server
  {
    libconfig::Config cfg;
    std::string py_config_file = sdk_path + "/" + "py.config";
    std::string py_bin, py_server;
    try {
      cfg.readFile(py_config_file.c_str());
    } catch (const libconfig::FileIOException &fioex) {
      std::cerr << "Could not open " << py_config_file << " for reading." << std::endl;
      return JANUS_OPEN_ERROR;
    } catch (const libconfig::ParseException &pex) {
      std::cerr << "Could not parse " << py_config_file << "." << std::endl;
      return JANUS_PARSE_ERROR;
    }

    try {
      libconfig::Setting &root = cfg.getRoot();

      libconfig::Setting &py_param = root["py_params"];
      py_param.lookupValue("py_env_dir", py_env_dir);
      py_env_dir = sdk_path + "/" + py_env_dir;
      setenv("PYTHONPATH", py_env_dir.c_str(), 1);

      py_param.lookupValue("py_bin", py_bin);
      py_param.lookupValue("py_server", py_server);

      // start python server
      std::map<std::string, std::string> params;
      params["py_bin"]    = py_bin;
      params["py_server"] = sdk_path + "/" + py_server;
      //janus_error status = pyServer.spawnPyWorker( params );
      janus_error status = JANUS_SUCCESS;
      if (status != JANUS_SUCCESS) return status;

      libconfig::Setting &cluster_param = root["py_params"]["cluster_params"];
      cluster_param.lookupValue("module", cluster_module);
      cluster_param.lookupValue("func", cluster_func);

      libconfig::Setting &tracker_param = root["py_params"]["tracker_params"];
      tracker_param.lookupValue("module", tracker_module);
      tracker_param.lookupValue("func", tracker_func);
    } catch (const libconfig::SettingNotFoundException &nfex) {
      std::cerr << "Could not find setting in py.config file." << std::endl;
      return JANUS_UNKNOWN_ERROR;
    }
  }
  ///////  Done with pyserver config


  if (algorithm != "noinit") {
    // Initialize face detection
    libconfig::Config cfg;
    std::string face_detect_config_file = sdk_path + "/face-detect.config";
    try {
      cfg.readFile(face_detect_config_file.c_str());
    } catch(const libconfig::FileIOException &fioex) {
      std::cerr << "Could not open " << face_detect_config_file << " for reading." << std::endl;
      return JANUS_OPEN_ERROR;
    } catch (const libconfig::ParseException &pex) {
      std::cerr << "Could not parse " << face_detect_config_file << "." << std::endl;
      return JANUS_PARSE_ERROR;
    }

    std::string yolo_cfgfile, yolo_weightfile;

    try {
      cfg.lookupValue("yolo_cfg", yolo_cfgfile);
      cfg.lookupValue("yolo_model", yolo_weightfile);

      yolo_cfgfile = sdk_path + "/" + yolo_cfgfile;
      yolo_weightfile = sdk_path + "/" + yolo_weightfile;

      cfg.lookupValue("yolo_prob_thresh", face_detect_probabiltiy_threshold);
      cfg.lookupValue("yolo_nms_thresh", face_detect_nms_threshold);
    } catch (const libconfig::SettingNotFoundException &nfex) {
      std::cerr << "Could not find setting in face-detect.config file." << std::endl;
      return JANUS_UNKNOWN_ERROR;
    }

    // TODO -- MUST use GPU advised by gpu_dev
    init_yolo(yolo_cfgfile.c_str(), yolo_weightfile.c_str());


    pose_detector.initialize(sdk_path, gpu_dev);

  }

  return JANUS_SUCCESS;
}


bool is_image(janus_association assoc) {
  return assoc.media.data.size() == 1;
}


float sign(float num) {
  return num < 0 ? -1 : 1;
}

void power_normization(std::vector<float> &fv, float pow) {

  std::vector<float> abs_fv(fv.size()), sgn_fv(fv.size());
  //   a. Create array of signs (i.e. either +1 or -1 entries)
  std::transform(fv.begin(), fv.end(), sgn_fv.begin(), sign);

  //   b. Create array of absolute values
  vsAbs(fv.size(), &(fv[0]), &(abs_fv[0]));

  //   c. Raise absolute array to power
  vsPowx(fv.size(), &(abs_fv[0]), pow, &(fv[0]));

  //   d. Finally multiply elementwise by sign vector
  vsMul(fv.size(), &(fv[0]), &(sgn_fv[0]), &(fv[0]));

}

void power_normization(float *fv, size_t n, float pow) {

  float *abs_fv = new float[n];
  float *sgn_fv = new float[n];

  //   a. Create array of signs (i.e. either +1 or -1 entries)
  std::transform(fv, fv+n, sgn_fv, sign);

  //   b. Create array of absolute values
  vsAbs(n, fv, abs_fv);

  //   c. Raise absolute array to power
  vsPowx(n, abs_fv, pow, fv);

  //   d. Finally multiply elementwise by sign vector
  vsMul(n, fv, sgn_fv, fv);

  delete [] abs_fv;
  delete [] sgn_fv;
}


featv_t do_feature_extraction(cv::Mat image) {

  int socket_fd = cnnServer.getSocket();

  char command = 'p';

  size_t bytes = write(socket_fd, &command, sizeof(char));
  if (bytes == -1) {
    perror("write socket");
  }

  bytes += CnnServer::writeMat(socket_fd, image);


  janus_error status;
  CnnServer::readData(socket_fd, &status, sizeof(status));

  featv_t feature_vector;
  if (status == JANUS_SUCCESS) {
    CnnServer::readVector(socket_fd, feature_vector);
  } else {
    perror("Failure in Featex");
  }
  close(socket_fd);

  // Finally power normalize feature
  power_normization(feature_vector, 0.65);


  // Now to PCA Transformation
  //  (a) Subtract PCA Mean
  std::cout << "About to do PCA transformation" << std::endl;
  size_t raw_feat_dim = backupPCATransform.cols;
  size_t transformed_feat_dim = backupPCATransform.rows;

  cblas_saxpy(raw_feat_dim, -1.0f, reinterpret_cast<const float*>(backupPCAMean.data), 1, &(feature_vector[0]), 1);
  //  (b) Matrix-vector Multiply
  featv_t transformed_feature_vector(transformed_feat_dim);
  cblas_sgemv(CblasRowMajor, CblasNoTrans, transformed_feat_dim, raw_feat_dim, 1.0f, reinterpret_cast<const float*>(backupPCATransform.data), raw_feat_dim, &(feature_vector[0]), 1, 0.0f, &(transformed_feature_vector[0]), 1);

  power_normization(transformed_feature_vector, 0.65);

  return transformed_feature_vector;
}

std::vector<featv_t> do_feature_extraction_batch(std::vector<cv::Mat> images) {

  int socket_fd = cnnServer.getSocket();

  char command = 'P'; // Use cap P for batch

  size_t bytes = write(socket_fd, &command, sizeof(char));
  if (bytes == -1) {
    perror("write socket");
  }

  // First write number of images
  int num_images = images.size();
  bytes = write(socket_fd, &num_images, sizeof(num_images));
  if (bytes == -1) {
    perror("write socket");
  }

  for (int i =0; i < num_images; ++i) {
    bytes += CnnServer::writeMat(socket_fd, images[i]);
  }

  janus_error status;
  CnnServer::readData(socket_fd, &status, sizeof(status));

  if (status != JANUS_SUCCESS) {
    perror("Failure in Featex");
  }

  std::vector<featv_t> feature_vectors;
  for (int i = 0; i < num_images; ++i) {
    featv_t feature_vector;
    std::cout << "trying to read batch vector (" << i << "/" << num_images << ")" << std::endl;
    CnnServer::readVector(socket_fd, feature_vector);

    std::cout << "Starting to process vector" << std::endl;
    // Finally power normalize feature
    power_normization(feature_vector, 0.65);

    // Now to PCA Transformation
    //  (a) Subtract PCA Mean
    std::cout << "About to do PCA transformation" << std::endl;
    size_t raw_feat_dim = backupPCATransform.cols;
    size_t transformed_feat_dim = backupPCATransform.rows;

    featv_t transformed_feature_vector(transformed_feat_dim);

    cblas_saxpy(raw_feat_dim, -1.0f, reinterpret_cast<const float*>(backupPCAMean.data), 1, &(feature_vector[0]), 1);
    //  (b) Matrix-vector Multiply
    cblas_sgemv(CblasRowMajor, CblasNoTrans, transformed_feat_dim, raw_feat_dim, 1.0f, reinterpret_cast<const float*>(backupPCATransform.data), raw_feat_dim, &(feature_vector[0]), 1, 0.0f, &(transformed_feature_vector[0]), 1);

    power_normization(transformed_feature_vector, 0.65);

    std::cout << "Pushing vector and moving on" << std::endl;
    feature_vectors.push_back(transformed_feature_vector);
  }

  close(socket_fd);

  std::cout << "Returning from extract batch on client side" << std::endl;
  return feature_vectors;
}

void extract_features_debug(int frame_num,
                            const janus_template_role role,
                            janus_association cur_association,
                            std::vector<featv_t> &featv_list,
                            std::vector<janus_error> &status_list,
                            pthread_mutex_t mtx,
                            cv::Mat &croppedImg,
                            cv::Mat &rendFrontalImg,
                            cv::Mat &rendHalfProfileImg,
                            cv::Mat &rendFullProfileImg,
                            cv::Mat &alignedImg,
                            float &yaw,
                            std::vector<cv::Point2f> &landmarks,
                            float &confidence)
{
  // First convert Janus media object into OpenCV object
  janus_error status;

  cv::Mat origImage;

  status = preproc.convertJanusMediaImageToOpenCV(cur_association.media, frame_num, origImage);
  if (status != JANUS_SUCCESS) {
    pthread_mutex_lock(&mtx);
    status_list.push_back(status);
    featv_t empty_fv;
    featv_list.push_back(empty_fv);
    pthread_mutex_unlock(&mtx);
    return;
  }

  // Determine pose
  janus_attributes metadata_attributes = cur_association.metadata.track[frame_num];
  int face_type = pose_detector.detect_pose(origImage, metadata_attributes.face_x, metadata_attributes.face_y, metadata_attributes.face_width, metadata_attributes.face_height);

  std::cout << "Found face type: " << face_type << std::endl;

  // Now run preprocessing (i.e. landmarks + alignment + rendering)
  status = preproc.process_withrender_debug(origImage, cur_association.metadata.track[frame_num], face_type, croppedImg, rendFrontalImg, rendHalfProfileImg, rendFullProfileImg, alignedImg, yaw, landmarks, confidence);

  if (status != JANUS_SUCCESS) {
    pthread_mutex_lock(&mtx);
    status_list.push_back(status);
    featv_t empty_fv;
    featv_list.push_back(empty_fv);
    pthread_mutex_unlock(&mtx);
    return;
  }

  // Now send preprocessed images to CNN Featex Server
  // For Pooling we do the following:
  //    If landmark worked:  pooled_feat = (1.0*real_aligned_feat + 0.3*rend_hp_feat + 0.3*rend_fp_feat + 0.3*rend_fr_feat)/(1+0.3+0.3+0.3)
  //     (note: if no rend-frontal-feat available, remove the term from numerator+denominator)
  //    If no landmarks: pooled_feat = cropped_image_feat
  featv_t pooled_feature_vector;

  std::cout << "Done with preproc, now on to featex" << std::endl;
  if (!croppedImg.empty()) {
    // Send over cropped image for feature extraction
    std::cout << "Only have cropped image, so doing featex of cropped image" << std::endl;

    try {
      pooled_feature_vector = do_feature_extraction(croppedImg);
    } catch (const std::exception &e) {
      std::cout << "Error: in feature extraction. Exception = " << e.what() << std::endl;
      pthread_mutex_lock(&mtx);
      status_list.push_back(JANUS_UNKNOWN_ERROR);
      featv_t empty_fv;
      featv_list.push_back(empty_fv);
      pthread_mutex_unlock(&mtx);
      return;
    }
  } else {
    if (alignedImg.empty() || rendHalfProfileImg.empty() || rendFullProfileImg.empty()) {
      std::cout << "Error: aligned or rendHalf or rendFull is empty -- shouldn't be the case given crop was empty(!)" << std::endl;
      pthread_mutex_lock(&mtx);
      status_list.push_back(status);
      featv_t empty_fv;
      featv_list.push_back(empty_fv);
      pthread_mutex_unlock(&mtx);
      return;
    }

    std::vector<cv::Mat> images;
    images.push_back(alignedImg);
    images.push_back(rendHalfProfileImg);
    images.push_back(rendFullProfileImg);

    float render_weight = 1.0f/3.0f;
    float denom = 1.0f + render_weight*2;
    if (!rendFrontalImg.empty()) {
      images.push_back(rendFrontalImg);
      denom += render_weight;
    }

    std::cout << "About to call feature extraction batch on rendered images" << std::endl;
    auto start_time = std::chrono::steady_clock::now();
    std::vector<featv_t> feature_vectors;

    try {
      feature_vectors = do_feature_extraction_batch(images);
    } catch (const std::exception &e) {
      std::cout << "Error: in feature extraction(batch). Exception = " << e.what() << std::endl;
      pthread_mutex_lock(&mtx);
      status_list.push_back(JANUS_UNKNOWN_ERROR);
      featv_t empty_fv;
      featv_list.push_back(empty_fv);
      pthread_mutex_unlock(&mtx);
      return;
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
    std::cout << "Batch featex took: " << duration.count() << " ms" << std::endl;

    std::cout << "About to pool rendered features" << std::endl;

    pooled_feature_vector = feature_vectors[0]; // (alignedImg has a weight of 1.0 in weighted average)
    size_t feat_dim = feature_vectors[0].size();

    if (feat_dim == 0) {
      std::cout << "Error: CNN Returned feature vector of dimension 0!" << std::endl;
      pthread_mutex_lock(&mtx);
      status_list.push_back(status);
      featv_t empty_fv;
      featv_list.push_back(empty_fv);
      pthread_mutex_unlock(&mtx);
      return;
    }

    for (int i = 1; i < feature_vectors.size(); ++i) {
      if (feat_dim != feature_vectors[i].size()) {
	std::cout << "Error: CNN Returned feature vectors of different dimensions [" << feat_dim << "] vs [" << feature_vectors[i].size() << "]" << std::endl;
	pthread_mutex_lock(&mtx);
	status_list.push_back(status);
	featv_t empty_fv;
	featv_list.push_back(empty_fv);
	pthread_mutex_unlock(&mtx);
	return;
      }

      cblas_saxpy(feat_dim, render_weight, &(feature_vectors[i][0]), 1, &(pooled_feature_vector[0]), 1);
    }
    cblas_sscal(feat_dim, 1.0f/denom, &(pooled_feature_vector[0]), 1);

    // Finally do power-norm on pooled feature
    power_normization(pooled_feature_vector, 0.65);

    std::cout << "Done with rendered pooling" << std::endl;
  }

  // And then add it to return list of feature vectors
  if (pooled_feature_vector.size() != 0)
  {
    std::cout << "Adding poooled feature to list" << std::endl;
    int pstatus = pthread_mutex_lock(&mtx);
    if (pstatus != 0) perror("Could not take lock");

    featv_list.push_back(pooled_feature_vector);
    status_list.push_back(JANUS_SUCCESS);

    pstatus = pthread_mutex_unlock(&mtx);
    if (pstatus != 0) perror("Could not release lock");

    std::cout << "Done adding pooled feature to list" << std::endl;
  }
}

void extract_features(int frame_num, const janus_template_role role, janus_association cur_association, std::vector<featv_t> &featv_list, std::vector<janus_error> &status_list, pthread_mutex_t mtx)
{
  // First convert Janus media object into OpenCV object
  janus_error status;

  cv::Mat origImage;

  status = preproc.convertJanusMediaImageToOpenCV(cur_association.media, frame_num, origImage);
  if (status != JANUS_SUCCESS) {
    pthread_mutex_lock(&mtx);
    status_list.push_back(status);
    featv_t empty_fv;
    featv_list.push_back(empty_fv);
    pthread_mutex_unlock(&mtx);
    return;
  }

  // Determine pose
  janus_attributes metadata_attributes = cur_association.metadata.track[frame_num];
  int face_type = pose_detector.detect_pose(origImage, metadata_attributes.face_x, metadata_attributes.face_y, metadata_attributes.face_width, metadata_attributes.face_height);

  std::cout << "Found face type: " << face_type << std::endl;

  // Now run preprocessing (i.e. landmarks + alignment + rendering)
  cv::Mat croppedImg, alignedImg, rendFrontalImg, rendHalfProfileImg, rendFullProfileImg;
  float yaw;
  status = preproc.process_withrender(origImage, cur_association.metadata.track[frame_num], face_type, croppedImg, rendFrontalImg, rendHalfProfileImg, rendFullProfileImg, alignedImg, yaw);

  if (status != JANUS_SUCCESS) {
    pthread_mutex_lock(&mtx);
    status_list.push_back(status);
    featv_t empty_fv;
    featv_list.push_back(empty_fv);
    pthread_mutex_unlock(&mtx);
    return;
  }

  // Now send preprocessed images to CNN Featex Server
  // For Pooling we do the following:
  //    If landmark worked:  pooled_feat = (1.0*real_aligned_feat + 0.3*rend_hp_feat + 0.3*rend_fp_feat + 0.3*rend_fr_feat)/(1+0.3+0.3+0.3)
  //     (note: if no rend-frontal-feat available, remove the term from numerator+denominator)
  //    If no landmarks: pooled_feat = cropped_image_feat
  featv_t pooled_feature_vector;

  std::cout << "Done with preproc, now on to featex" << std::endl;
  if (!croppedImg.empty()) {
    // Send over cropped image for feature extraction
    std::cout << "Only have cropped image, so doing featex of cropped image" << std::endl;

    try {
      pooled_feature_vector = do_feature_extraction(croppedImg);
    } catch (const std::exception &e) {
      std::cout << "Error: in feature extraction. Exception = " << e.what() << std::endl;
      pthread_mutex_lock(&mtx);
      status_list.push_back(JANUS_UNKNOWN_ERROR);
      featv_t empty_fv;
      featv_list.push_back(empty_fv);
      pthread_mutex_unlock(&mtx);
      return;
    }
  } else {
    if (alignedImg.empty() || rendHalfProfileImg.empty() || rendFullProfileImg.empty()) {
      std::cout << "Error: aligned or rendHalf or rendFull is empty -- shouldn't be the case given crop was empty(!)" << std::endl;
      pthread_mutex_lock(&mtx);
      status_list.push_back(status);
      featv_t empty_fv;
      featv_list.push_back(empty_fv);
      pthread_mutex_unlock(&mtx);
      return;
    }

    std::vector<cv::Mat> images;
    images.push_back(alignedImg);
    images.push_back(rendHalfProfileImg);
    images.push_back(rendFullProfileImg);

    float render_weight = 1.0f/3.0f;
    float denom = 1.0f + render_weight*2;
    if (!rendFrontalImg.empty()) {
      images.push_back(rendFrontalImg);
      denom += render_weight;
    }

    std::cout << "About to call feature extraction batch on rendered images" << std::endl;
    auto start_time = std::chrono::steady_clock::now();
    std::vector<featv_t> feature_vectors;

    try {
      feature_vectors = do_feature_extraction_batch(images);
    } catch (const std::exception &e) {
      std::cout << "Error: in feature extraction(batch). Exception = " << e.what() << std::endl;
      pthread_mutex_lock(&mtx);
      status_list.push_back(JANUS_UNKNOWN_ERROR);
      featv_t empty_fv;
      featv_list.push_back(empty_fv);
      pthread_mutex_unlock(&mtx);
      return;
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
    std::cout << "Batch featex took: " << duration.count() << " ms" << std::endl;

    std::cout << "About to pool rendered features" << std::endl;

    pooled_feature_vector = feature_vectors[0]; // (alignedImg has a weight of 1.0 in weighted average)
    size_t feat_dim = feature_vectors[0].size();

    if (feat_dim == 0) {
      std::cout << "Error: CNN Returned feature vector of dimension 0!" << std::endl;
      pthread_mutex_lock(&mtx);
      status_list.push_back(status);
      featv_t empty_fv;
      featv_list.push_back(empty_fv);
      pthread_mutex_unlock(&mtx);
      return;
    }

    for (int i = 1; i < feature_vectors.size(); ++i) {
      if (feat_dim != feature_vectors[i].size()) {
	std::cout << "Error: CNN Returned feature vectors of different dimensions [" << feat_dim << "] vs [" << feature_vectors[i].size() << "]" << std::endl;
	pthread_mutex_lock(&mtx);
	status_list.push_back(status);
	featv_t empty_fv;
	featv_list.push_back(empty_fv);
	pthread_mutex_unlock(&mtx);
	return;
      }

      cblas_saxpy(feat_dim, render_weight, &(feature_vectors[i][0]), 1, &(pooled_feature_vector[0]), 1);
    }
    cblas_sscal(feat_dim, 1.0f/denom, &(pooled_feature_vector[0]), 1);

    // Finally do power-norm on pooled feature
    power_normization(pooled_feature_vector, 0.65);

    std::cout << "Done with rendered pooling" << std::endl;
  }

  // And then add it to return list of feature vectors
  if (pooled_feature_vector.size() != 0)
  {
    std::cout << "Adding poooled feature to list" << std::endl;
    int pstatus = pthread_mutex_lock(&mtx);
    if (pstatus != 0) perror("Could not take lock");

    featv_list.push_back(pooled_feature_vector);
    status_list.push_back(JANUS_SUCCESS);

    pstatus = pthread_mutex_unlock(&mtx);
    if (pstatus != 0) perror("Could not release lock");

    std::cout << "Done adding pooled feature to list" << std::endl;
  }

}


janus_error do_pooling(std::vector< featv_t > &featv_list, std::vector< janus_error > &status_list, int &out_count, featv_t &out_pooled_featv) {

  // Make sure we got features back
  if (featv_list.size() == 0)
    return JANUS_FAILURE_TO_ENROLL;

  // Time for media pooling--Average together all features (1) First from videos; (2) Then from everything
  // Find first valid feature, and find total # of features
  size_t first_valid_idx = -1;
  size_t valid_cnt = 0;

  for (size_t i = 0; i < featv_list.size(); ++i) {
    if (status_list[i] == JANUS_SUCCESS) {
      if (first_valid_idx == -1)
	first_valid_idx = i;
      valid_cnt += 1;
    }
  }

  // Set this so later on we can verify we had at least one feature vector to pool
  out_count = valid_cnt;

  if (valid_cnt == 0) {
    return JANUS_SUCCESS;;
  }


  std::cout << "About to pool [" << valid_cnt << "] different images from template together" << std::endl;
  size_t feat_dim = featv_list[first_valid_idx].size();
  out_pooled_featv = featv_list[first_valid_idx];


  if (valid_cnt > 1) {
    // Add up all the vectors
    for (size_t i = first_valid_idx + 1; i < featv_list.size(); ++i) {
      if (status_list[i] == JANUS_SUCCESS) {
	if (feat_dim != featv_list[i].size()) {
	  std::cout << "Feature dimensions mismatch: [" << feat_dim << "] vs [" << featv_list[i].size() << "]" << std::endl;
	  return JANUS_UNKNOWN_ERROR;
	}
	cblas_saxpy(feat_dim, 1.0f, &(featv_list[i][0]), 1, &(out_pooled_featv[0]), 1);
      }
    }

    // Now divide by count
    cblas_sscal(feat_dim, 1.0f/((float)valid_cnt), &(out_pooled_featv[0]), 1);

    // Finally do power norm one last time
    power_normization(out_pooled_featv, 0.65);

    std::cout << "Done with pooling for now, moving on." << std::endl;
  }

  return JANUS_SUCCESS;
}

JANUS_EXPORT janus_error janus_debug(janus_association &association,
                        const janus_template_role role,
                        cv::Mat &cropped,
                        cv::Mat &rend_fr,
                        cv::Mat &rend_hp,
                        cv::Mat &rend_fp,
                        cv::Mat &aligned,
                        float &yaw,
                        std::vector<cv::Point2f> &landmarks,
                        float &confidence)
{
  // Need to send images to CNN Server in big batch; to do this set up background worker threads
  std::vector<std::thread>  workers;

  std::vector< featv_t > featv_list;
  std::vector< janus_error > status_list;
  pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;


  // For each image in tempalte, startup a new thread

    if (is_image(association)) {
      workers.push_back(std::thread(extract_features_debug, /*frameNum=*/0, role, std::ref(association),
				    std::ref(featv_list), std::ref(status_list), std::ref(mtx),
            std::ref(cropped), std::ref(rend_fr), std::ref(rend_hp), std::ref(rend_fp), std::ref(aligned), std::ref(yaw), std::ref(landmarks), std::ref(confidence)));

      // Because landmark detector is not thread safe, we have to do a join here, to do sequentially
      // TODO: create "thread pool" of landmark detectors that is actually a "process pool", i.e. forked processes
      for (std::thread& t : workers)
	t.join();
      workers.clear();
    } else { // video
      // Okay either we have a collection of frames, or we have all frames from a video
      std::vector< featv_t > video_featv_list;
      std::vector< janus_error > video_status_list;
      pthread_mutex_t video_mtx = PTHREAD_MUTEX_INITIALIZER;
      for (int frame_num = 0; frame_num < association.media.data.size(); ++frame_num) {
	workers.push_back(std::thread(extract_features_debug, frame_num, role, std::ref(association),
				      std::ref(video_featv_list), std::ref(video_status_list), std::ref(video_mtx),
              std::ref(cropped), std::ref(rend_fr), std::ref(rend_hp), std::ref(rend_fp), std::ref(aligned), std::ref(yaw), std::ref(landmarks), std::ref(confidence)));

	for (std::thread& t : workers)
	  t.join();
	workers.clear();
      }

      // Do pooling for video seperately, then add to template feat list
      int valid_cnt;
      featv_t pooled_video_feat;
      janus_error status = do_pooling(video_featv_list, video_status_list, valid_cnt, pooled_video_feat);
      if (status == JANUS_SUCCESS && valid_cnt > 0) {
	// If we succesfully pooled video features, let's add them to the list of template feature vectors
	//  (Which will undergo second round of pooling below

	int pstatus = pthread_mutex_lock(&mtx);
	if (pstatus != 0) perror("Could not take lock");

	featv_list.push_back(pooled_video_feat);
	status_list.push_back(JANUS_SUCCESS);

	pstatus = pthread_mutex_unlock(&mtx);
	if (pstatus != 0) perror("Could not release lock");
      }
    }


  // Wait for all threads to finish
  for (std::thread& t : workers)
    t.join();

  return JANUS_SUCCESS;
}

JANUS_EXPORT janus_error janus_create_template(std::vector<janus_association> &associations,
                                               const janus_template_role role,
                                               janus_template &template_)
{
  // Need to send images to CNN Server in big batch; to do this set up background worker threads
  std::vector<std::thread>  workers;

  std::vector< featv_t > featv_list;
  std::vector< janus_error > status_list;
  pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

  std::cout << "Starting template creation" << std::endl;

  // For each image in tempalte, startup a new thread
  for (auto &&cur_association : associations) {
    if (is_image(cur_association)) {
      workers.push_back(std::thread(extract_features, /*frameNum=*/0, role, std::ref(cur_association),
				    std::ref(featv_list), std::ref(status_list), std::ref(mtx)));

      // Because landmark detector is not thread safe, we have to do a join here, to do sequentially
      // TODO: create "thread pool" of landmark detectors that is actually a "process pool", i.e. forked processes
      for (std::thread& t : workers)
	t.join();
      workers.clear();
    } else { // video
      // Okay either we have a collection of frames, or we have all frames from a video
      std::vector< featv_t > video_featv_list;
      std::vector< janus_error > video_status_list;
      pthread_mutex_t video_mtx = PTHREAD_MUTEX_INITIALIZER;
      for (int frame_num = 0; frame_num < cur_association.media.data.size(); ++frame_num) {
	workers.push_back(std::thread(extract_features, frame_num, role, std::ref(cur_association),
				      std::ref(video_featv_list), std::ref(video_status_list), std::ref(video_mtx)));

	for (std::thread& t : workers)
	  t.join();
	workers.clear();
      }

      // Do pooling for video seperately, then add to template feat list
      int valid_cnt;
      featv_t pooled_video_feat;
      janus_error status = do_pooling(video_featv_list, video_status_list, valid_cnt, pooled_video_feat);
      if (status == JANUS_SUCCESS && valid_cnt > 0) {
	// If we succesfully pooled video features, let's add them to the list of template feature vectors
	//  (Which will undergo second round of pooling below

	int pstatus = pthread_mutex_lock(&mtx);
	if (pstatus != 0) perror("Could not take lock");

	featv_list.push_back(pooled_video_feat);
	status_list.push_back(JANUS_SUCCESS);

	pstatus = pthread_mutex_unlock(&mtx);
	if (pstatus != 0) perror("Could not release lock");
      }
    }
  }

  // Wait for all threads to finish
  for (std::thread& t : workers)
    t.join();

  std::cout << "Done with featex for this template, moving on..." << std::endl;

  // Make sure we got features back
  if (featv_list.size() == 0)
    return JANUS_FAILURE_TO_ENROLL;


  int valid_cnt;
  featv_t pooled_feature_vector;
  janus_error status = do_pooling(featv_list, status_list, valid_cnt, pooled_feature_vector);

  if (status != JANUS_SUCCESS) return status;
  if (valid_cnt == 0) return JANUS_SUCCESS;


  // Now construct template using the computed feature vectors
  template_ = new janus_template_type;
  template_->pooled_feat = pooled_feature_vector;


  std::cout << "Done with template." << std::endl;
  return JANUS_SUCCESS;
}

// JANUS_EXPORT janus_error janus_create_template_debug(std::vector<janus_association> &associations,
//                                                const janus_template_role role,
//                                                janus_template &template_,
//                                                std::vector<cv::Mat> &out_cropped,
//                                              	 std::vector<cv::Mat> &out_rend_fr,
//                                              	 std::vector<cv::Mat> &out_rend_hp,
//                                              	 std::vector<cv::Mat> &out_rend_fp,
//                                              	 std::vector<cv::Mat> &out_aligned,
//                                              	 float *out_yaw,
//                                              	 std::vector<std::vector<cv::Point2f>> &out_landmarks,
//                                              	 float *out_confidence)
// {
//   // Need to send images to CNN Server in big batch; to do this set up background worker threads
//   std::vector<std::thread>  workers;
//
//   std::vector< featv_t > featv_list;
//   std::vector< janus_error > status_list;
//   pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
//
//   std::cout << "Starting template creation" << std::endl;
//
//   int i = 0;
//
//   // For each image in tempalte, startup a new thread
//   for (auto &&cur_association : associations) {
//     if (is_image(cur_association)) {
//       workers.push_back(std::thread(extract_features_debug, /*frameNum=*/0, role, std::ref(cur_association),
// 				    std::ref(featv_list), std::ref(status_list), std::ref(mtx),
//             std::ref(out_cropped[i]), std::ref(out_rend_fr[i]), std::ref(out_rend_hp[i]), std::ref(out_rend_fp[i]),
//             std::ref(out_aligned[i])));// std::ref(out_yaw[i]), std::ref(out_landmarks[i]), std::ref(out_confidence[i])));
//       // Because landmark detector is not thread safe, we have to do a join here, to do sequentially
//       // TODO: create "thread pool" of landmark detectors that is actually a "process pool", i.e. forked processes
//       for (std::thread& t : workers)
//         t.join();
//       workers.clear();
//     } else { // video
//       // Okay either we have a collection of frames, or we have all frames from a video
//       std::vector< featv_t > video_featv_list;
//       std::vector< janus_error > video_status_list;
//       pthread_mutex_t video_mtx = PTHREAD_MUTEX_INITIALIZER;
//       for (int frame_num = 0; frame_num < cur_association.media.data.size(); ++frame_num) {
// 	workers.push_back(std::thread(extract_features, frame_num, role, std::ref(cur_association),
// 				      std::ref(video_featv_list), std::ref(video_status_list), std::ref(video_mtx)));
//
// 	for (std::thread& t : workers)
// 	  t.join();
// 	workers.clear();
//       }
//
//       // Do pooling for video seperately, then add to template feat list
//       int valid_cnt;
//       featv_t pooled_video_feat;
//       janus_error status = do_pooling(video_featv_list, video_status_list, valid_cnt, pooled_video_feat);
//       if (status == JANUS_SUCCESS && valid_cnt > 0) {
// 	// If we succesfully pooled video features, let's add them to the list of template feature vectors
// 	//  (Which will undergo second round of pooling below
//
// 	int pstatus = pthread_mutex_lock(&mtx);
// 	if (pstatus != 0) perror("Could not take lock");
//
// 	featv_list.push_back(pooled_video_feat);
// 	status_list.push_back(JANUS_SUCCESS);
//
// 	pstatus = pthread_mutex_unlock(&mtx);
// 	if (pstatus != 0) perror("Could not release lock");
//       }
//     }
//     i++;
//   }
//
//   // Wait for all threads to finish
//   for (std::thread& t : workers)
//     t.join();
//
//   std::cout << "Done with featex for this template, moving on..." << std::endl;
//
//   // Make sure we got features back
//   if (featv_list.size() == 0)
//     return JANUS_FAILURE_TO_ENROLL;
//
//
//   int valid_cnt;
//   featv_t pooled_feature_vector;
//   janus_error status = do_pooling(featv_list, status_list, valid_cnt, pooled_feature_vector);
//
//   if (status != JANUS_SUCCESS) return status;
//   if (valid_cnt == 0) return JANUS_SUCCESS;
//
//
//   // Now construct template using the computed feature vectors
//   template_ = new janus_template_type;
//   template_->pooled_feat = pooled_feature_vector;
//
//
//   std::cout << "Done with template." << std::endl;
//   return JANUS_SUCCESS;
// }

int max_index(float *a, int n)
{
  if(n <= 0) return -1;
  int i, max_i = 0;
  float max = a[0];
  for(i = 1; i < n; ++i){
    if(a[i] > max){
      max = a[i];
      max_i = i;
    }
  }
  return max_i;
}


void convert_box_to_track( box *boxes, float **probs, size_t num_boxes, size_t yolo_num_classes, double frame_number,
                           std::vector<cv::Rect> &boundingboxes, /* initial bounding boxes for tracking */
                           const janus_media &media, const size_t min_face_size, std::vector<janus_track> &tracks )
{
  for(size_t ibox = 0; ibox < num_boxes; ++ibox) {
    int i_class = max_index(probs[ibox], yolo_num_classes);
    float prob = probs[ibox][i_class];
    if ( prob < face_detect_probabiltiy_threshold )
      continue;

    box b = boxes[ibox];
    int left  = (b.x-b.w/2.)*media.width;
    int right = (b.x+b.w/2.)*media.width;
    int top   = (b.y-b.h/2.)*media.height;
    int bot   = (b.y+b.h/2.)*media.height;

    if(left < 0) left = 0;
    if(right > media.width-1) right = media.width-1;
    if(top < 0) top = 0;
    if(bot > media.height-1) bot = media.height-1;

    if (right - left < min_face_size || bot - top < min_face_size)
      continue;

    janus_track cur_track;
    cur_track.gender = cur_track.age = cur_track.skin_tone = std::numeric_limits<float>::quiet_NaN();
    cur_track.frame_rate = 0;

    janus_attributes cur_attribute;
    cur_attribute.face_x = left;
    cur_attribute.face_y = top;
    cur_attribute.face_width = right - left;
    cur_attribute.face_height = bot - top;

    cur_attribute.right_eye_x = cur_attribute.right_eye_y = cur_attribute.left_eye_x = cur_attribute.left_eye_y = cur_attribute.nose_base_x = cur_attribute.nose_base_y = cur_attribute.face_yaw = std::numeric_limits<float>::quiet_NaN();
    cur_attribute.frame_number = frame_number;

    cur_track.track.push_back(cur_attribute);
    cur_track.detection_confidence = prob;
    tracks.push_back(cur_track);

    // Initial bounding boxes for tracking
    cv::Rect rect(left, top, right-left, bot-top);
    boundingboxes.push_back( rect );
  }

}

void convert_box_to_track( const int minx, const int maxx, const int miny, const int maxy, double frame_number, const int track_id,
                           const size_t min_face_size, janus_track &track )
{
    if (maxx - minx < min_face_size || maxy - miny < min_face_size)
        return;

    // current track attribute
    janus_attributes cur_attribute;
    cur_attribute.face_x = minx;
    cur_attribute.face_y = miny;
    cur_attribute.face_width  = maxx - minx;
    cur_attribute.face_height = maxy - miny;

    cur_attribute.right_eye_x = cur_attribute.right_eye_y = cur_attribute.left_eye_x = cur_attribute.left_eye_y = cur_attribute.nose_base_x = cur_attribute.nose_base_y = cur_attribute.face_yaw = std::numeric_limits<float>::quiet_NaN();
    cur_attribute.frame_number = frame_number;

    track.track.push_back(cur_attribute);
}

void init_numpy()
{
  import_array();
}


JANUS_EXPORT janus_error janus_detect(const janus_media &media,
                                      const size_t min_face_size,
                                      std::vector<janus_track> &tracks)
{
  // Don't support video trackign currently
  if (media.data.size() == 0)
    return JANUS_FAILURE_TO_DETECT;

  // Detection for a single image frame
  if (media.data.size() == 1) {
    std::cout << "janus_detect for a single image." << std::endl;

    // Create an OpenCV Mat wrapper around data, then convert to an IplImage wrapper
    cv::Mat img_mat;
    janus_error status = preproc.convertJanusMediaImageToOpenCV(media, /*frame_num =*/0, img_mat);
    if (status != JANUS_SUCCESS) return status;

    IplImage img_ipl = img_mat;

    // Now setup+run yolo
    box *out_boxes = NULL;
    float **out_probs = NULL;
    size_t out_num_boxes;
    size_t yolo_num_classes;

    test_yolo_janus(face_detect_probabiltiy_threshold, face_detect_nms_threshold, &img_ipl, &out_boxes, &out_probs, &out_num_boxes, &yolo_num_classes);

    // Now convert boxes from yolo to janus format
    for(size_t ibox = 0; ibox < out_num_boxes; ++ibox) {
      int i_class = max_index(out_probs[ibox], yolo_num_classes);
      float prob = out_probs[ibox][i_class];
      if ( prob < face_detect_probabiltiy_threshold )
        continue;
      box b = out_boxes[ibox];
      int left  = (b.x-b.w/2.)*media.width;
      int right = (b.x+b.w/2.)*media.width;
      int top   = (b.y-b.h/2.)*media.height;
      int bot   = (b.y+b.h/2.)*media.height;

      if(left < 0) left = 0;
      if(right > media.width-1) right = media.width-1;
      if(top < 0) top = 0;
      if(bot > media.height-1) bot = media.height-1;

      if (right - left < min_face_size || bot - top < min_face_size)
	continue;

      janus_track cur_track;
      cur_track.gender = cur_track.age = cur_track.skin_tone = std::numeric_limits<float>::quiet_NaN();
      cur_track.frame_rate = 0;

      janus_attributes cur_attribute;
      cur_attribute.face_x = left;
      cur_attribute.face_y = top;
      cur_attribute.face_width = right - left;
      cur_attribute.face_height = bot - top;

      cur_attribute.right_eye_x = cur_attribute.right_eye_y = cur_attribute.left_eye_x = cur_attribute.left_eye_y = cur_attribute.nose_base_x = cur_attribute.nose_base_y = cur_attribute.face_yaw = std::numeric_limits<float>::quiet_NaN();

      cur_track.track.push_back(cur_attribute);
      cur_track.detection_confidence = prob;
      tracks.push_back(cur_track);
    }

    // Finally, free yolo memory
    free_yolo_mem(&out_boxes, &out_probs);
  } else {
    std::cout << "janus_detect for video frames." << std::endl;

    // Handle video now
    // Detection for the first image frame
    // Create an OpenCV Mat wrapper around data, then convert to an IplImage wrapper
    cv::Mat img_mat;
    janus_error status = preproc.convertJanusMediaImageToOpenCV(media, /*frame_num =*/0, img_mat);
    if (status != JANUS_SUCCESS) return status;

    IplImage img_ipl = img_mat;

    // Now setup+run yolo
    box *out_boxes = NULL;
    float **out_probs = NULL;
    size_t out_num_boxes;
    size_t yolo_num_classes;
    std::vector<cv::Rect> initial_bounding_boxes;

    test_yolo_janus(face_detect_probabiltiy_threshold, face_detect_nms_threshold, &img_ipl, &out_boxes, &out_probs, &out_num_boxes, &yolo_num_classes);

    // Now convert boxes from yolo to janus format and save it for tracking
    convert_box_to_track( out_boxes, out_probs, out_num_boxes, yolo_num_classes, /*frame_number*/ 0, initial_bounding_boxes, media, min_face_size, tracks );

    std::cout << "There are " << initial_bounding_boxes.size() << " initial bounding boxes are detected." << std::endl;

    if (!py_init) {
      Py_Initialize();
      init_numpy();
      py_init = true;
    }

    // Process the remaining frames
    if (media.data.size() != 1 && initial_bounding_boxes.size() != 0) {

      int rows         = media.width;
      int cols         = media.height;
      int frames       = media.data.size();
      int ND           = 3;
      npy_intp dims[3] = { frames, cols, rows };

      PyObject *pName, *pModule, *pFunc, *pVideoBuff;
      PyArrayObject *np_arr, *np_ret;

      // prepare video buffer data
      uint8_t *video_buf = new uint8_t [frames*rows*cols];
      if (video_buf == NULL) {
        std::cerr << "Out of memory for tracking." << std::endl;
        return JANUS_FAILURE_TO_DETECT;
      }

      for (size_t i = 0; i < frames; ++i) {
        cv::Mat video_frame;
        status = preproc.convertJanusMediaImageToOpenCV(media, i, video_frame);
        if (status != JANUS_SUCCESS) {
          delete [] video_buf;
          return JANUS_FAILURE_TO_DETECT;
        }
        if (media.color_space == JANUS_BGR24) { cv::cvtColor(video_frame, video_frame, CV_RGB2GRAY, 1); }
        memcpy( video_buf+i*rows*cols, video_frame.data, sizeof(uint8_t)*rows*cols );
      }

      // Convert array from c++ to python
      pVideoBuff = PyArray_SimpleNewFromData(ND, dims, NPY_UINT8, reinterpret_cast<void*>(video_buf));
      if (pVideoBuff == NULL) {
        std::cerr << "Converting args from c++ to python for " << tracker_func << "failed." << std::endl;
        goto FAIL_NP_ARRAY;
      }
      np_arr = reinterpret_cast<PyArrayObject*>(pVideoBuff);

      // Load module
      pName = PyString_FromString( tracker_module.c_str() );
      if (pName == NULL) {
        std::cerr << "Could not find python module " << tracker_module << "." << std::endl;
        goto FAIL_NAME;
      }

      pModule = PyImport_Import( pName );
      if (pModule == NULL) {
        std::cerr << "Could not import python module " << tracker_module << "." << std::endl;
        goto FAIL_IMPORT;
      }

      // Load function from module
      pFunc = PyObject_GetAttrString( pModule, tracker_func.c_str() );
      if (pFunc == NULL) {
        std::cerr << "Could not load python function " << tracker_func << " from module " << tracker_module << "." << std::endl;
        goto FAIL_GETATTR;
      }
      if (!PyCallable_Check(pFunc)) {
        std::cerr << "Python function " << tracker_func << " from module " << tracker_module << " is not callable." << std::endl;
        goto FAIL_CALLABLE;
      }

      // For each initial bounding box
      for (size_t track_id = 0; track_id < initial_bounding_boxes.size(); ++track_id) {
        // get init bounding box
        PyObject *pBoxBuff, *pReturn;
        PyArrayObject *np_box_arr;
        npy_intp box_dims[1] = {4};
        int boxes[4] = {initial_bounding_boxes[track_id].x,
                        initial_bounding_boxes[track_id].x + initial_bounding_boxes[track_id].width,
                        initial_bounding_boxes[track_id].y,
                        initial_bounding_boxes[track_id].y + initial_bounding_boxes[track_id].height };

        pBoxBuff = PyArray_SimpleNewFromData(1, box_dims, NPY_INT32, reinterpret_cast<void*>(&boxes));
        if (pBoxBuff == NULL) {
          std::cerr << "Converting box information from c++ to python failed." << std::endl;
          continue;
        }
        np_box_arr = reinterpret_cast<PyArrayObject*>(pBoxBuff);

        // Calling python function
        std::cout << "calling alien_tracker" << std::endl;
        pReturn = PyObject_CallFunctionObjArgs(pFunc, np_arr, np_box_arr, NULL);
        if (pReturn == NULL) {
          std::cerr << "Executing python function " << tracker_func << " failed." << std::endl;
          goto FAIL_CALL;
        }
        if (!PyList_Check(pReturn)) {
          std::cerr << "Python function " << tracker_func << " did not return list." << std::endl;
          goto FAIL_LIST_CHECK;
        }

        if (PyList_Size(pReturn) != frames - 1) {
          std::cerr << "Returned list with wrong length for " << tracker_func << "." << std::endl;
          goto FAIL_DIM;
        }

        for (size_t frame_num = 1; frame_num < PyList_Size(pReturn)+1; ++frame_num) {
          PyObject *bbox = PyList_GetItem(pReturn, frame_num-1);
          if (!PyList_Check(bbox))
            continue;

          if (PyList_Size(bbox) != 4)
            continue;

          long minx = PyInt_AsLong(PyList_GetItem(bbox, 0));
          long maxx = PyInt_AsLong(PyList_GetItem(bbox, 1));
          long miny = PyInt_AsLong(PyList_GetItem(bbox, 2));
          long maxy = PyInt_AsLong(PyList_GetItem(bbox, 3));

          convert_box_to_track( minx, maxx, miny, maxy, frame_num, track_id,
                                min_face_size, tracks[track_id] );

          Py_XDECREF(bbox);
        }

        Py_XDECREF(pBoxBuff);
        Py_XDECREF(pReturn);
      }

  // clean up
FAIL_DIM:
FAIL_LIST_CHECK:
FAIL_CALL:
FAIL_CALLABLE:
  Py_XDECREF(pFunc);
FAIL_GETATTR:
  Py_XDECREF(pModule);
FAIL_IMPORT:
  Py_XDECREF(pName);
FAIL_NAME:
  Py_XDECREF(pVideoBuff);
FAIL_NP_ARRAY:
  delete [] video_buf;
FAIL_C_ARRAY:
  if (PyErr_Occurred()) PyErr_Print();
    }

    // Finally, free yolo memory
    free_yolo_mem(&out_boxes, &out_probs);
  }

  return JANUS_SUCCESS;
}


JANUS_EXPORT janus_error janus_create_template(const janus_media &media,
                                               const janus_template_role role,
                                               std::vector<janus_template> &templates,
                                               std::vector<janus_track> &tracks)
{
  janus_error status = JANUS_UNKNOWN_ERROR;

  // tracking the media
  size_t min_face_size = 40;

  status = janus_detect(media, min_face_size, tracks);
  if (status != JANUS_SUCCESS) {
    std::cerr << "Tracking failed." << std::endl;
    return status;
  }

  // extract features for each face (track)
  for (size_t i = 0; i < tracks.size(); ++i) {
    std::vector<janus_association> associations;
    janus_template cur_template;
    janus_track    cur_track = tracks[i];

    for (size_t j = 0; j < cur_track.track.size(); ++j) {
      int frame_id = cur_track.track[j].frame_number;

      janus_media cur_media;
      cur_media.width       = media.width;
      cur_media.height      = media.height;
      cur_media.step        = media.step;
      cur_media.color_space = media.color_space;
      cur_media.data.push_back( media.data[frame_id] );

      janus_track metadata;
      metadata.detection_confidence = cur_track.detection_confidence;
      metadata.gender     = cur_track.gender;
      metadata.age        = cur_track.age;
      metadata.skin_tone  = cur_track.skin_tone;
      metadata.frame_rate = cur_track.frame_rate;

      metadata.track.push_back( cur_track.track[j] );

      janus_association association;
      association.media    = cur_media;
      association.metadata = metadata;

      associations.push_back( association );
    }

    status = janus_create_template(associations, role, cur_template);
    if (status != JANUS_SUCCESS) {
      std::cerr << "Create template for media failed." << std::endl;
    }

    templates.push_back( cur_template );
  }

  return status;
}


JANUS_EXPORT janus_error janus_serialize_template(const janus_template &template_,
                                                  std::ostream &stream)
{
  // Make sure template is non-empty
  if (template_->pooled_feat.size() == 0) {
    size_t num_feats = 0;
    stream.write(reinterpret_cast<const char*>(&num_feats), sizeof(size_t));
    return JANUS_SUCCESS;
  }

  // Make sure feature dimension is non-empty
  size_t feat_dim = template_->pooled_feat.size();

  std::cout << "\n\n\tWriting out template, feat_dim = " << feat_dim << std::endl;

  if (feat_dim != 2048)
    return JANUS_UNKNOWN_ERROR;

  // First write out the feature vector dimension
  stream.write(reinterpret_cast<const char*>(&feat_dim), sizeof(size_t));

  // Now write out feature vector
  stream.write(reinterpret_cast<const char*>(&(template_->pooled_feat[0])), sizeof(float)*feat_dim);

  return JANUS_SUCCESS;
}

JANUS_EXPORT janus_error janus_deserialize_template(janus_template &template_,
                                                    std::istream &stream)
{
  template_ = new janus_template_type;

  // First read the feature vector dimension
  size_t feat_dim;
  stream.read(reinterpret_cast<char*>(&feat_dim), sizeof(size_t));

  if (feat_dim != 2048) // ?? If feat dim is not what we expect, return empty template
    return JANUS_SUCCESS;

  // Check if we are reading an an empty (FTE) template; if so, return immediately
  if (feat_dim == 0)
    return JANUS_SUCCESS;

  // Now read in feature vector
  featv_t feat(feat_dim);
  stream.read(reinterpret_cast<char*>(&feat[0]), sizeof(float)*feat_dim);

  if (!stream) {
    std::cout << "Error reading template stream; Tried to read in [" << feat_dim*sizeof(float) << "] bytes";
    std::cout << ", but could only read [" << stream.gcount() << "] bytes instead" << std::endl;
    return JANUS_UNKNOWN_ERROR;
  }


  template_->pooled_feat = feat;

  return JANUS_SUCCESS;
}


JANUS_EXPORT janus_error janus_delete_template(janus_template &template_)
{
  delete template_;
  return JANUS_SUCCESS;
}


void centerMatrixRow(float *A, const int m, const int n);
void normalizeMatrixRow(float *A, const int m, const int n);


JANUS_EXPORT janus_error janus_create_gallery(const std::vector<janus_template> &templates,
                                              const std::vector<janus_template_id> &ids,
                                              janus_gallery &gallery)
{
  // Sanity Check -- we should have an id for each template
  if (templates.size() != ids.size())
    return JANUS_UNKNOWN_ERROR;

  std::cout << "Creating gallery, # Templates = " << templates.size() << std::endl;

  // Reserve gallery size, then add each template to gallery
  gallery = new janus_gallery_type;
  size_t num_templates = templates.size();

  gallery->template_ids.reserve(num_templates);

  // Need to find first non-empty template and count number of non-empty templates
  gallery->feat_dim = 0;
  size_t nonempty_template_count = 0;
  for (size_t i = 0; i < templates.size(); ++i) {
    if (templates[i]->pooled_feat.size() > 0) {
      gallery->feat_dim =  templates[i]->pooled_feat.size(); // Doens't hurt to reset this, should be same every time through loop
      nonempty_template_count += 1;
    }
  }

  // Verify we have at least one non-empty template
  if (nonempty_template_count == 0) {
    std::cout << "ERROR: gallery only contains empty templates" << std::endl;
    return JANUS_UNKNOWN_ERROR;
  }

  std::cout << "# Non-empty templates = " << nonempty_template_count << std::endl;

  gallery->feats = new float[gallery->feat_dim * nonempty_template_count];

  // Now ready to do memcpys
  for (size_t i = 0; i < templates.size(); ++i) {
    if (templates[i]->pooled_feat.size() == 0)
      continue; // Skip over empty templates

    size_t cur_idx = gallery->template_ids.size();
    memcpy(gallery->feats + cur_idx*gallery->feat_dim, &(templates[i]->pooled_feat[0]), sizeof(float)*gallery->feat_dim);
    gallery->template_ids.push_back(ids[i]);
  }


  // Finally, subtract mean and normalize gallery feats as a pre-computation (otherwise would need to do this for each call to search)
  centerMatrixRow(gallery->feats, nonempty_template_count, gallery->feat_dim);
  normalizeMatrixRow(gallery->feats, nonempty_template_count, gallery->feat_dim);


  std::cout << "Done with creating gallery" << std::endl;

  return JANUS_SUCCESS;
}


JANUS_EXPORT janus_error janus_serialize_gallery(const janus_gallery &gallery,
                                                 std::ostream &stream)
{
  // Make sure gallery is non-empty
  size_t num_templates = gallery->template_ids.size();

  std::cout << "Serializing gallery with " << num_templates << " non-empty templates." << std::endl;

  if (num_templates == 0)
    return JANUS_FAILURE_TO_SERIALIZE;

  // First write out the number of templates
  stream.write(reinterpret_cast<const char*>(&num_templates), sizeof(size_t));

  // Write out the feature dimension
  stream.write(reinterpret_cast<const char*>(&gallery->feat_dim), sizeof(size_t));

  // Write out template ID's
  stream.write(reinterpret_cast<const char*>(&gallery->template_ids[0]), sizeof(size_t) * num_templates);

  // Write out the features
  stream.write(reinterpret_cast<const char*>(gallery->feats), sizeof(float) * gallery->feat_dim * num_templates);

  return JANUS_SUCCESS;
}

JANUS_EXPORT janus_error janus_deserialize_gallery(janus_gallery &gallery,
                                                   std::istream &stream)
{
  gallery = new janus_gallery_type;

  // First, read in number of templates
  size_t num_templates;
  stream.read(reinterpret_cast<char*>(&num_templates), sizeof(size_t));

  // Now read in the feature dimension
  stream.read(reinterpret_cast<char*>(&gallery->feat_dim), sizeof(size_t));

  // Now read in the template ID's
  gallery->template_ids.resize(num_templates);
  stream.read(reinterpret_cast<char*>(&gallery->template_ids[0]), sizeof(size_t) * num_templates);

  // Now read in the features
  gallery->feats = new float[gallery->feat_dim * num_templates];
  stream.read(reinterpret_cast<char*>(gallery->feats), sizeof(float) * gallery->feat_dim * num_templates);

  return JANUS_SUCCESS;

}

JANUS_EXPORT janus_error janus_prepare_gallery(janus_gallery &gallery)
{
  return JANUS_SUCCESS;
}

JANUS_EXPORT janus_error janus_gallery_insert(janus_gallery &gallery,
                                              const janus_template &template_,
                                              const janus_template_id id)
{
  (void)gallery;
  (void)template_;
  (void)id;

  return JANUS_NOT_IMPLEMENTED;
}

JANUS_EXPORT janus_error janus_gallery_remove(janus_gallery &gallery,
                                              const janus_template_id id)
{
  (void)gallery;
  (void)id;

  return JANUS_NOT_IMPLEMENTED;
}

JANUS_EXPORT janus_error janus_delete_gallery(janus_gallery &gallery)
{
  delete gallery;
  return JANUS_SUCCESS;
}



////////////////////////////////////////////////////////////////////////////////////
////   Temporary spot for some correlation methods
////////////////////////////////////////////////////////////////////////////////////

double meanRow(const float A[], const int row, const int m, const int n) {
  double mean = 0;
  int index = row*n;
  for (int j=0; j < n; ++j) {
    mean += A[index + j];
  }
  mean = mean/n;
  return mean;
}

void centerMatrixRow(float *A, const int m, const int n) {
  for (int i=0; i<m; ++i) {
    float mean = meanRow(A, i, m, n);
    int index = i*n;
    for (int j=0; j < n; ++j) {
      A[index+j] = A[index+j] - mean;
    }
  }
}

void normalizeMatrixRow(float *A, const int m, int n) {
  for (int i=0; i<m; ++i) {
    float norm = cblas_snrm2(n, A+i*n, 1);
    int index = i*n;
    for (int j=0; j<n; ++j) {
      A[index+j] = A[index+j] / norm;
    }
  }
}

void fast_correlation_normboth(float *A, float *B, float *C, const int m, const int n, const int k) {
  // Matrix A has dimensions m * k
  // Matrix B has dimensions n * k
  // Matrix C has dimensions m * n


  centerMatrixRow(A, m, k);
  centerMatrixRow(B, n, k);

  normalizeMatrixRow(A, m, k);
  normalizeMatrixRow(B, n, k);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0, A, k, B, k, 0.0, C, n);
}

void fast_correlation_normprobe(float *A, float *B, float *C, const int m, const int n, const int k) {
  // Matrix A has dimensions m * k
  // Matrix B has dimensions n * k
  // Matrix C has dimensions m * n


  centerMatrixRow(B, n, k);
  normalizeMatrixRow(B, n, k);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0, A, k, B, k, 0.0, C, n);
}

void compute_similarity(float *corrMatrix, int galDim, int probeDim, std::vector<size_t> gal_template_sizes, int numTemplates, std::vector<float> &similarity)  {

  int start = 0;
  for (int tp=0; tp<numTemplates; ++tp) {
    int corrLength = probeDim * gal_template_sizes[tp];

    if (corrLength == 0) {
      similarity[tp] = -1;
    } else if (corrLength == 1) {
      similarity[tp] = (corrMatrix+start)[0];
    } else {
      int indexMax = cblas_isamax( corrLength, corrMatrix+start, 1 );
      float rawmax = (corrMatrix+start)[indexMax];

      float alpha = 10.0;
      float numerator = 0;
      float denominator = 0;
      for (size_t i = 0; i < corrLength; ++i) {
	numerator += (corrMatrix+start)[i] * exp( alpha * ( (corrMatrix+start)[i] - rawmax) );
	denominator += exp( alpha * ( (corrMatrix+start)[i] - rawmax) );
      }
      similarity[tp] = numerator / denominator;

    }
    start += corrLength;
  }
}

bool orderSearchResults(std::pair<janus_template_id, float> a, std::pair<janus_template_id, float> b) {
  return a.second > b.second;
}

////////////////////////////////////////////////////////////////////////////////////
////   END OF Temporary spot for some correlation methods
////////////////////////////////////////////////////////////////////////////////////


JANUS_EXPORT janus_error janus_verify(const janus_template &reference,
                                      const janus_template &verification,
                                      double &similarity)
{
  // Already applied PCA + powernorm to features when creating template; also did pooling
  // Each template only contains a single feature vector, so only need to do corr. coef for one pair of vectors
  if (reference->pooled_feat.size() == 0 || verification->pooled_feat.size() == 0) {
    std::cout << "ERROR: janus_verify called with empty templates" << std::endl;
    return JANUS_UNKNOWN_ERROR;
  }
  if (reference->pooled_feat.size() != verification->pooled_feat.size()) {
    std::cout << "ERROR: verification feature dimensions do not match" << std::endl;
    std::cout << "\treference->pooled_feat.size() = " << reference->pooled_feat.size() << std::endl;
    std::cout << "verification->pooled_feat.size() = " << verification->pooled_feat.size() << std::endl;
    return JANUS_UNKNOWN_ERROR;
  }


  float corr_score;
  fast_correlation_normboth(&(reference->pooled_feat[0]), &(verification->pooled_feat[0]), &corr_score, 1, 1, reference->pooled_feat.size());

  similarity = corr_score;

  return JANUS_SUCCESS;
}


JANUS_EXPORT janus_error janus_search(const janus_template &probe,
                                      const janus_gallery &gallery,
                                      const size_t num_requested_returns,
                                      std::vector<janus_template_id> &template_ids,
                                      std::vector<double> &similarities)
{
  size_t num_gal_templates = gallery->template_ids.size();
  if (num_gal_templates == 0) return JANUS_UNKNOWN_ERROR;

  size_t feat_dim = gallery->feat_dim;

  //if (probe->pooled_feat.size() == 0) return JANUS_UNKNOWN_ERROR;
  if (probe->pooled_feat.size() == 0) return JANUS_SUCCESS;

  float *correlationMatrix = new float[gallery->template_ids.size()];

  fast_correlation_normprobe(gallery->feats, &(probe->pooled_feat[0]), correlationMatrix, gallery->template_ids.size(), 1, feat_dim);

  // Construct array of all (score,id) pairs
  std::vector< std::pair<janus_template_id, float> > search_results;
  for (int i=0; i < num_gal_templates; ++i) {
    search_results.push_back(std::make_pair(gallery->template_ids[i], correlationMatrix[i]));
  }
  // Sort by similarity score
  std::sort(search_results.begin(), search_results.end(), orderSearchResults);

  // Return only the top N results
  for (size_t i = 0; i < num_requested_returns && i < search_results.size(); ++i) {
    template_ids.push_back(search_results[i].first);
    similarities.push_back(search_results[i].second);
  }

  delete [] correlationMatrix;

  return JANUS_SUCCESS;

}


JANUS_EXPORT janus_error janus_cluster(const std::vector<janus_template> &templates,
                                       const size_t hint,
                                       std::vector<cluster_pair> &clusters)
{
  janus_error status = JANUS_UNKNOWN_ERROR;

  if (templates.size() == 0)
    return status;


//  PyGILState_STATE gstate;
//  gstate = PyGILState_Ensure();
  Py_Initialize();
  init_numpy();

  int len;
  long long *label;
  PyObject *pArray, *pName, *pModule, *pFunc, *pReturn;
  PyArrayObject *np_arr, *np_ret;

  // Load features into memory
  int ND  = 2;
  int row = templates.size();
  int col = 0;

  for (int i = 0; i < templates.size(); ++i) {
    if (templates[i]->pooled_feat.size() > 0) {
      col = templates[i]->pooled_feat.size();
      break;
    }
  }

  if (col == 0) {
    std::cout << "No templates are non-empty, for clustering" << std::endl;
    return JANUS_UNKNOWN_ERROR;
  }

  npy_intp dims[2]   = { row, col };
  float *c_arr = new float [row*col];

  if (c_arr == NULL) {
    std::cerr << "Out of memory for clustering." << std::endl;
    goto FAIL_C_ARRAY;
  }

  for (size_t i = 0; i < row; ++i) {
    if ( templates[i]->pooled_feat.size() != col ) {
      for (size_t j = 0; j < col; ++j)
        c_arr[i*col+j] = 0;
    } else {
      for (size_t j = 0; j < col; ++j)
        c_arr[i*col+j] = templates[i]->pooled_feat[j];
    }
  }

  // Convert array from c++ to python
  pArray = PyArray_SimpleNewFromData(ND, dims, NPY_FLOAT, reinterpret_cast<void*>(c_arr));
  if (pArray == NULL) {
    std::cerr << "Converting args from c++ to python for " << cluster_func << "failed." << std::endl;
    goto FAIL_NP_ARRAY;
  }
  np_arr = reinterpret_cast<PyArrayObject*>(pArray);

  // Load module
  pName = PyString_FromString( cluster_module.c_str() );
  if (pName == NULL) {
    std::cerr << "Could not find python module " << cluster_module << "." << std::endl;
    goto FAIL_NAME;
  }

  pModule = PyImport_Import( pName );
  if (pModule == NULL) {
    std::cerr << "Could not import python module " << cluster_module << "." << std::endl;
    goto FAIL_IMPORT;
  }

  // Load function from module
  pFunc = PyObject_GetAttrString( pModule, cluster_func.c_str() );
  if (pFunc == NULL) {
    std::cerr << "Could not load python function " << cluster_func << " from module " << cluster_module << "." << std::endl;
    goto FAIL_GETATTR;
  }
  if (!PyCallable_Check(pFunc)) {
    std::cerr << "Python function " << cluster_func << " from module " << cluster_module << " is not callable." << std::endl;
    goto FAIL_CALLABLE;
  }

  // Calling python function
  pReturn = PyObject_CallFunctionObjArgs(pFunc, np_arr, NULL);
  if (pReturn == NULL) {
    std::cerr << "Executing python function " << cluster_func << " failed." << std::endl;
    goto FAIL_CALL;
  }
  if (!PyArray_Check(pReturn)) {
    std::cerr << "Python function " << cluster_func << " did not return an array." << std::endl;
    goto FAIL_ARRAY_CHECK;
  }

  np_ret = reinterpret_cast<PyArrayObject*>(pReturn);
  if (PyArray_NDIM(np_ret) != ND - 1) {
    std::cerr << "Returned array with wrong dimension for " << cluster_func << "." << std::endl;
    goto FAIL_DIM;
  }

  // Convert output value from python to c++
  clusters.clear();
  len   = PyArray_SHAPE(np_ret)[0];
  label = reinterpret_cast<long long*>(PyArray_DATA(np_ret));

  for (size_t i = 0; i < len; ++i) {
    cluster_pair value = std::make_pair(label[i], 1.00);
    clusters.push_back( value );
  }

  status = JANUS_SUCCESS;

  // clean up
FAIL_DIM:
FAIL_ARRAY_CHECK:
  Py_DECREF(pReturn);
FAIL_CALL:
FAIL_CALLABLE:
  Py_DECREF(pFunc);
FAIL_GETATTR:
  Py_DECREF(pModule);
FAIL_IMPORT:
  Py_DECREF(pName);
FAIL_NAME:
  Py_DECREF(pArray);
FAIL_NP_ARRAY:
  delete [] c_arr;
FAIL_C_ARRAY:
  if (PyErr_Occurred()) PyErr_Print();


//  PyGILState_Release(gstate);
//  Py_Finalize();

  return status;
}



JANUS_EXPORT janus_error janus_finalize()
{
  if (g_is_python_up) pythonFinalize();

  Py_Finalize();

  return JANUS_SUCCESS;
}

janus_error load_pca(std::string pca_file, cv::Mat &out_PCATransform, cv::Mat &out_PCAMean) {
  // Now read in PCA transformation matrix + feature mean

  std::ifstream stream(pca_file);

  size_t pca_rows, pca_cols;
  stream.read(reinterpret_cast<char*>(&pca_rows), sizeof(size_t));
  stream.read(reinterpret_cast<char*>(&pca_cols), sizeof(size_t));

  out_PCATransform = cv::Mat(pca_rows, pca_cols, CV_32FC1);
  stream.read(reinterpret_cast<char*>(out_PCATransform.data), sizeof(float) * pca_rows * pca_cols);

  size_t raw_feat_dim = pca_cols;
  out_PCAMean = cv::Mat(1, raw_feat_dim, CV_32FC1);
  stream.read(reinterpret_cast<char*>(out_PCAMean.data), sizeof(float) * raw_feat_dim);

  return JANUS_SUCCESS;
}

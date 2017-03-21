#include "CnnPoseDetection.hpp"

#include <boost/shared_ptr.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <libconfig.h++>

janus_error CnnPoseDetection::initialize(std::string sdk_path, const int gpu_dev)
{
  libconfig::Config cfg;
  std::string pose_config_file = sdk_path + "/pose.config";

  try {
    cfg.readFile(pose_config_file.c_str());
  } catch(const libconfig::FileIOException &fioex) {
    std::cerr << "Could not open " << pose_config_file << " for reading." << std::endl;
    return JANUS_OPEN_ERROR;
  } catch (const libconfig::ParseException &pex) {
    std::cerr << "Could not parse " << pose_config_file << "." << std::endl;
    return JANUS_PARSE_ERROR;
  }

  std::string cnn_model_definition;
  std::string cnn_pretrained_weights;
  std::string cnn_feature_layer;

  try {
     cfg.lookupValue("cnn_model_definition", cnn_model_definition);
     cfg.lookupValue("cnn_pretrained_weights", cnn_pretrained_weights);
     cfg.lookupValue("cnn_feature_layer", cnn_feature_layer);
     cfg.lookupValue("target_width", target_width);
     cfg.lookupValue("target_height", target_height);
  } catch (const libconfig::SettingNotFoundException &nfex) {
    std::cerr << "Could not find setting in preproc.config file." << std::endl;
    return JANUS_UNKNOWN_ERROR;
  }

  cnn_model_definition = sdk_path + "/" + cnn_model_definition;
  cnn_pretrained_weights = sdk_path + "/" + cnn_pretrained_weights;
  m_feature_layer = cnn_feature_layer;

  // Set GPU mode
  caffe::Caffe::SetDevice(gpu_dev);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);

  // Initialize network weights
  pose_net = new caffe::Net<float>(cnn_model_definition, caffe::TEST);
  pose_net->CopyTrainedLayersFrom(cnn_pretrained_weights);

  // Figure out feature vector size
  const boost::shared_ptr< caffe::Blob<float> > feature_blob = pose_net->blob_by_name(m_feature_layer);
  cnn_feature_dimension_size = feature_blob->count();

  return JANUS_SUCCESS;
}


int CnnPoseDetection::detect_pose(cv::Mat img_in, int face_x, int face_y, int face_w, int face_h)
{
  if (img_in.empty()) {
    std::cout << "called extract() with empty image" << std::endl;
    return 0;
  }

  caffe::Blob<float>* input_layer = pose_net->input_blobs()[0];

  // Do special preprocessing required by pose  (crop + resize + grayscale)
  face_h = face_h*0.7;
  face_y = face_y + face_h * 0.3;

  if (face_x < 0)
    face_x = 0;
  if (face_x >= img_in.cols)
    face_x = img_in.cols-1;
  if (face_y < 0)
    face_y = 0;
  if (face_y >= img_in.rows)
    face_y = img_in.rows-1;
    

  if (face_x + face_w >= img_in.cols)
    face_w = img_in.cols - 1 - face_x;
  if (face_y + face_h >= img_in.rows)
    face_h = img_in.rows - 1 - face_y;

  if (face_w == 0 || face_h == 0)
    return 0;

  cv::Mat img_in_copy = img_in.clone();
  cv::Rect cropped_region = cv::Rect(face_x, face_y, face_w, face_h);
  cv::Mat cropped_img(img_in_copy, cropped_region);

  cv::Mat resized_img;
  cv::resize(cropped_img, resized_img, cv::Size(96,96), 0, 0, CV_INTER_CUBIC);

  cv::Mat grayscale_img;
  cv::cvtColor(resized_img, grayscale_img, CV_BGR2GRAY);

  int num_channels = grayscale_img.channels();

  input_layer->Reshape(/*batchsize=*/1, num_channels, grayscale_img.rows, grayscale_img.cols);
  int index = 0;
  for (int h = 0; h < grayscale_img.rows; ++h) {
    for (int w = 0; w < grayscale_img.cols; ++w) {
      input_layer->mutable_cpu_data()[ index ] = grayscale_img.at<uchar>(h, w);
      index++;
    }
  }

  // Done preprocessing, no feed into network


  // Now run CNN forward pass
  pose_net->Reshape();
  pose_net->ForwardPrefilled();

  // Now let's parse network output
  const boost::shared_ptr<caffe::Blob< float > > layer_blob = pose_net->blob_by_name(m_feature_layer);
  const float *layer_blob_data = layer_blob->cpu_data();

  std::cout << "\tPose regressxor: " << layer_blob_data[1] << ";\t(x90) = " << layer_blob_data[1]*90 << std::endl;

  if (layer_blob_data[1] * 90 <= 30.0 && layer_blob_data[1] * 90 >= -30.0)
    return 1;
  else if (layer_blob_data[1] * 90 < -30.0)
    return 2;
  else
    return 3;

}


#include <thread>
#include <chrono>

#include "CnnFeatex.hpp"

#include <boost/shared_ptr.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <libconfig.h++>

janus_error CnnFeatex::initialize(std::string sdk_path, const int gpu_dev)
{
  libconfig::Config cfg;
  std::string featex_config_file = sdk_path + "/featex.config";

  try {
    cfg.readFile(featex_config_file.c_str());
  } catch(const libconfig::FileIOException &fioex) {
    std::cerr << "Could not open " << featex_config_file << " for reading." << std::endl;
    return JANUS_OPEN_ERROR;
  } catch (const libconfig::ParseException &pex) {
    std::cerr << "Could not parse " << featex_config_file << "." << std::endl;
    return JANUS_PARSE_ERROR;
  }

  std::string train_mean_image;
  std::string cnn_model_definition;
  std::string cnn_pretrained_weights;
  std::string cnn_feature_layer;

  try {
     cfg.lookupValue("mean_image", train_mean_image);
     cfg.lookupValue("cnn_model_definition", cnn_model_definition);
     cfg.lookupValue("cnn_pretrained_weights", cnn_pretrained_weights);
     cfg.lookupValue("cnn_feature_layer", cnn_feature_layer);
  } catch (const libconfig::SettingNotFoundException &nfex) {
    std::cerr << "Could not find setting in preproc.config file." << std::endl;
    return JANUS_UNKNOWN_ERROR;
  }

  train_mean_image = sdk_path + "/" + train_mean_image;
  cnn_model_definition = sdk_path + "/" + cnn_model_definition;
  cnn_pretrained_weights = sdk_path + "/" + cnn_pretrained_weights;
  m_feature_layer = cnn_feature_layer;

  // Set GPU mode
  caffe::Caffe::SetDevice(gpu_dev);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);

  // Initialize network weights
  feature_extraction_net = new caffe::Net<float>(cnn_model_definition, caffe::TEST);
  feature_extraction_net->CopyTrainedLayersFrom(cnn_pretrained_weights);

  // Figure out feature vector size
  const boost::shared_ptr< caffe::Blob<float> > feature_blob = feature_extraction_net->blob_by_name(m_feature_layer);
  cnn_feature_dimension_size = feature_blob->count();

  // Load mean image
  caffe::BlobProto mean_image_proto;
  caffe::ReadProtoFromBinaryFile(train_mean_image.c_str(), &mean_image_proto);
  mean_image_blob.FromProto(mean_image_proto);


  m_initialized.store(true);


  return JANUS_SUCCESS;
}


std::vector<featv_t> CnnFeatex::extract_batch(std::vector<cv::Mat*> input_image_list)
{
  if (!m_initialized.load()) {
    while(1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      if (!m_initialized.load())
	break;
    }
  }

  if (input_image_list.size() == 0) {
    std::cout << "called extract_batch() with zero images" << std::endl;
    std::vector<featv_t> empty_list;
    return empty_list;
  }
  for (size_t i = 0; i < input_image_list.size(); ++i) {
    if (input_image_list[i]->empty()) {
      std::cout << "Called extract batch() with an empty image!" << std::cout;
      std::vector<featv_t> empty_list;
      return empty_list;
    }
  }

  caffe::Blob<float>* input_layer = feature_extraction_net->input_blobs()[0];

  // Convert input image list to format expected by caffe; also subtract mean image from input images
  CvImageListToBlob(input_image_list, mean_image_blob.height(), mean_image_blob.width(), mean_image_blob, input_layer);

  // Now run CNN forward pass
  feature_extraction_net->Reshape();
  feature_extraction_net->ForwardPrefilled();

  // Now let's get the features out of the CNN
  size_t numImages = input_image_list.size();
  std::vector<featv_t> feature_vector_list(numImages);
  for (size_t i=0; i < numImages; ++i) {
    // Pre-allocate vector of appropriate size
    std::vector<float> feature_vector(cnn_feature_dimension_size);

    // Grab pointer to appropriate layer inside of CNN, and point to appropriate image within layer
    const boost::shared_ptr<caffe::Blob< float > > layer_blob = feature_extraction_net->blob_by_name(m_feature_layer);
    const float *layer_blob_data = layer_blob->cpu_data() + layer_blob->offset(i);

    // Populate feature vector by memcpy'ing from network layer
    memcpy(&feature_vector[0], layer_blob_data, sizeof(float) * layer_blob->count() / numImages);

    // Finally add current feature vector to batch return value
    feature_vector_list[i] = feature_vector;
  }

  return feature_vector_list;
}

bool CnnFeatex::CvImageListToBlob(std::vector<cv::Mat*> image_list, int target_height, int target_width, const caffe::Blob<float> &mean_image_blob, caffe::Blob<float>* out_blob)
{
  int num_images = image_list.size();

  if (num_images ==0) {
    return false;
  }
  int num_channels = image_list[0]->channels();

  std::vector< cv::Mat > resized_image_list;
  cv::Size targetSize(target_width, target_height);
  for (auto it=image_list.begin(); it != image_list.end(); ++it) {
    cv::Mat resized_image;

    if (target_height > 0 && target_width > 0) {
      cv::resize(**it, resized_image, targetSize);
      resized_image_list.push_back(resized_image);
    } else {
      resized_image_list.push_back(**it);
    }
  }

  out_blob->Reshape(num_images, num_channels, resized_image_list[0].rows, resized_image_list[0].cols);

  int index = 0;
  for (int i=0; i < num_images; ++i) {
    int mindex = 0;
    if (num_channels > 1) {
      for (int c = 0; c < num_channels; ++c) {
	for (int h = 0; h < resized_image_list[i].rows; ++h) {
	  for (int w = 0; w < resized_image_list[i].cols; ++w) {
	    out_blob->mutable_cpu_data()[ index ] = resized_image_list[i].at<cv::Vec3b>(h, w)[c] - mean_image_blob.cpu_data()[ mindex ];
	    index++;
	    mindex++;
	  }
	}
      }
    } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
      for (int h = 0; h < resized_image_list[i].rows; ++h) {
	for (int w = 0; w < resized_image_list[i].cols; ++w) {
	  out_blob->mutable_cpu_data()[ index ] = resized_image_list[i].at<uchar>(h, w) - mean_image_blob.cpu_data()[ mindex ];
	  index++;
	  mindex++;
	}
      }
    }
  }
  return true;
}

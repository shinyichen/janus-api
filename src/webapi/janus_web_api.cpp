#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <boost/python.hpp>

#include "iarpa_janus.h"
#include "iarpa_janus_io.h"

char const* hello()
{
  return "hello";
}

char const* str2char(boost::python::object s)
{
  using namespace boost::python;
  const char* value = extract<char const*>(s);
  return value;
}

void initialize(char * sdk_path, char * temp_path)
{
  char *algorithm = "";
  int gpu_dev = 0;
  JANUS_ASSERT(janus_initialize(sdk_path, temp_path, algorithm, gpu_dev));
}

// create single template
double create_template(char * image_path, janus_attributes attributes)
{
  // create media from image path
  janus_media media;
  janus_load_media(image_path, media);

  // create janus_attributes, expects face_x, face_y, face_width, face_height to be set already
  attributes.right_eye_x = NAN;
  attributes.right_eye_y = NAN;
  attributes.left_eye_x = NAN;
  attributes.left_eye_y = NAN;
  attributes.nose_base_x = NAN;
  attributes.nose_base_y = NAN;
  attributes.face_yaw = NAN;
  attributes.forehead_visible = NAN;
  attributes.eyes_visible = NAN;
  attributes.nose_mouth_visible = NAN;
  attributes.indoor = NAN;
  attributes.frame_number = NAN;

  // create janus_track from janus_attributes
  // TODO what about detection_confidence?, gender?, age?, skin_tone?, frame_rate?
  janus_track track;
  track.track.push_back(attributes);

  // create janus_association from janus_media, janus_track
  janus_association association;
  association.media = media;
  association.metadata = track;
  std::vector<janus_association> associations;
  associations.push_back(association);

  // create template from association, role, template
  janus_template template1;
  janus_template_role role;
  janus_create_template(associations, role, template1);

  // create janus_assocation from media and track
  janus_free_media(media);

  return 0;

}

// int search(char *probes_list_file, char *gallery_list_file, char *gallery_file, char *num_requested_returns, char *candidate_list_file)
// {
//     bool verbose = false;
//     janus_search_helper(probes_list_file, gallery_list_file, gallery_file, atoi(num_requested_returns), candidate_list_file, verbose);
//
//     return EXIT_SUCCESS;
// }

void finalize()
{
  JANUS_ASSERT(janus_finalize());
}

BOOST_PYTHON_MODULE(janus_web_api)
{
    using namespace boost::python;
    def("hello", hello);
    def("initialize", initialize);
    // enum_<janus_color_space>("janus_color_space")
    //     .value("JANUS_GRAY8", JANUS_GRAY8)
    //     .value("JANUS_BGR24", JANUS_BGR24)
    //     ;
    class_<janus_attributes>("janus_attributes")
        .def_readwrite("face_x", &janus_attributes::face_x)
        .def_readwrite("face_y", &janus_attributes::face_y)
        .def_readwrite("face_width", &janus_attributes::face_width)
        .def_readwrite("face_height", &janus_attributes::face_height)
        .def_readwrite("right_eye_x", &janus_attributes::right_eye_x)
        .def_readwrite("right_eye_y", &janus_attributes::right_eye_y)
        .def_readwrite("left_eye_x", &janus_attributes::left_eye_x)
        .def_readwrite("left_eye_y", &janus_attributes::left_eye_y)
        .def_readwrite("nose_base_x", &janus_attributes::nose_base_x)
        .def_readwrite("nose_base_y", &janus_attributes::nose_base_y)
        .def_readwrite("face_yaw", &janus_attributes::face_yaw)
        .def_readwrite("forehead_visible", &janus_attributes::forehead_visible)
        .def_readwrite("eyes_visible", &janus_attributes::eyes_visible)
        .def_readwrite("nose_mouth_visible", &janus_attributes::nose_mouth_visible)
        .def_readwrite("indoor", &janus_attributes::indoor)
        .def_readwrite("frame_number", &janus_attributes::frame_number)
        ;

    def("create_template", create_template);
    // def("search", search);
    def("finalize", finalize);
}

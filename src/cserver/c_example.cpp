#include <fstream>
#include <iostream>
#include <sstream>
#include <csignal>
#include "crow_all.h"
#include "iarpa_janus_io.h"
#include "iarpa_janus.h"

// to compile
// c++ -Wall -std=c++1y -pedantic -Wextra c_example.cpp -I include -Llib -lboost_system (-lglaive) -o c_example.out
using namespace std;
using namespace crow;

void signalHandler( int signum ) {
   cout << "Finalize Janus" << endl;
   janus_finalize();
   exit(signum);

}

int main()
{
    crow::SimpleApp app;

    // initialize janus
    cout << "Initialize Janus" << endl;
    janus_initialize("/nfs/isicvlnas01/users/xpeng/projects/Janus/release/janus-isi-sdk-feb/", "/tmp/", "", 0);

    // finalize janus when program abort
    signal(SIGINT, signalHandler);

		CROW_ROUTE(app, "/search")
    .methods("POST"_method)
    ([](const request& req) {

      // json: list of images
      // [{image_path, face_x, face_y, face_width, face_height}, ...]
			auto arr = json::load(req.body); // req.body is string, arr is of type rvalue
			if (!arr)
				return response(400);

      string gallery_path = "/nfs/isicvlnas01/users/srawls/janus-dev/scratch/output-new/1N-gal-S1.gal";

      cout << "========== CServer: Process inputs ==============" << endl;

      int image_count = arr.size();
      vector<janus_association> associations;
      for (int i = 0; i < image_count; i ++) {
        auto img = arr[i];
        string image_path = json::dump(img["image_path"]);
        // remove double quotes from string
        image_path.erase(remove(image_path.begin(), image_path.end(), '\"' ), image_path.end());
        double face_x = img["face_x"].d();
        double face_y = img["face_y"].d();
        double face_width = img["face_width"].d();
        double face_height = img["face_height"].d();

        cout << image_path << ":" << to_string(face_x) << "," << to_string(face_y) << "," << to_string(face_width) << "," << to_string(face_height) << endl;

        // create media from image path
        janus_media media;
        janus_load_media(image_path, media);

        // create janus_attributes
  			janus_attributes attributes;
  			attributes.face_x = face_x;
  			attributes.face_y = face_y;
  			attributes.face_width = face_width;
  			attributes.face_height = face_height;
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

        // for each image, create an association
  		  associations.push_back(association);
      }

      cout << "========== CServer: Creating Template ==============" << endl;

		  // create template from associations, role, template
		  janus_template template1;
		  janus_template_role role;
		  int ct_res = janus_create_template(associations, role, template1);

      for (int i = 0; i < image_count; i ++) {
        janus_media media = associations[i].media;
        janus_free_media(media);
      }

      if (ct_res != 0)
        return response("janus_create_template failed");

      // ====== search ==========

      cout << "========== CServer: Searching ==============" << endl;

      // create janus_gallery from gallery path
      cout << gallery_path.c_str() << endl;
      ifstream gallery_stream(gallery_path.c_str(), ios::in | ios::binary);
      janus_gallery gallery = NULL;
      janus_deserialize_gallery(gallery, gallery_stream);

      vector<janus_template_id> return_template_ids;
      vector<double> similarities;
      int s_res = janus_search(template1, gallery, 50, return_template_ids, similarities);
      janus_delete_gallery(gallery);
      janus_delete_template(template1);

      if (s_res != 0)
        return response("janus_search failed");



      json::wvalue result;
      for (int i = 0; i < return_template_ids.size(); i++) {
        result[i]["template_id"] = return_template_ids[i];
        result[i]["similarity"] = similarities[i];
      }

      return response(result);

    });

    app.port(8080).run();
}

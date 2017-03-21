#include <sstream>
#include "crow_all.h"
#include "iarpa_janus.h"

// to compile
// c++ -Wall -std=c++1y -pedantic -Wextra c_example.cpp -I include -Llib -lboost_system (-lglaive) -o c_example.out
using namespace std;
using namespace crow;

int main()
{
    crow::SimpleApp app;

    CROW_ROUTE(app, "/initialize")
    ([]() {
        if (janus_initialize("/nfs/isicvlnas01/users/xpeng/projects/Janus/release/janus-isi-sdk-feb/", "/tmp/", "", 0) == 0) {
					return "successful";
				} else {
					return "failed";
				}
    });


		CROW_ROUTE(app, "/search")
    .methods("POST"_method)
    ([](const request& req) {

      // json: list of images
      // [{image_path, face_x, face_y, face_width, face_height}, ...]
			auto arr = json::load(req.body); // req.body is string, arr is of type rvalue
			if (!arr)
				return response(400);

      string gallery_path = "/nfs/isicvlnas01/users/srawls/janus-dev/scratch/output-new/1N-gal-S1.gal";

      int image_count = arr.size();

      for (int i = 0; i < image_count; i ++) {
        auto img = arr[i];
        string image_path = json::dump(img["image_path"]);
        int face_x = img["face_x"].i();
        int face_y = img["face_y"].i();
        int face_width = img["face_width"].i();
        int face_height = img["face_height"].i();

        cout << image_path << ":" << to_string(face_x) << "," << to_string(face_y) << "," << to_string(face_width) << "," << to_string(face_height) << endl;

        // create media from image path
      //   janus_media media;
      //   janus_load_media(image_path, media);
      //
      //   // create janus_attributes
  		// 	janus_attributes attributes;
  		// 	attributes.face_x = face_x;
  		// 	attributes.face_y = face_y;
  		// 	attributes.face_width = face_width;
  		// 	attributes.face_height = face_height;
  		// 	attributes.right_eye_x = NAN;
  		//   attributes.right_eye_y = NAN;
  		//   attributes.left_eye_x = NAN;
  		//   attributes.left_eye_y = NAN;
  		//   attributes.nose_base_x = NAN;
  		//   attributes.nose_base_y = NAN;
  		//   attributes.face_yaw = NAN;
  		//   attributes.forehead_visible = NAN;
  		//   attributes.eyes_visible = NAN;
  		//   attributes.nose_mouth_visible = NAN;
  		//   attributes.indoor = NAN;
  		//   attributes.frame_number = NAN;
      //
      //   // create janus_track from janus_attributes
  		//   // TODO what about detection_confidence?, gender?, age?, skin_tone?, frame_rate?
  		//   janus_track track;
  		//   track.track.push_back(attributes);
      //
      //   // create janus_association from janus_media, janus_track
  		//   janus_association association;
  		//   association.media = media;
  		//   association.metadata = track;
  		//   std::vector<janus_association> associations;
      //
      //   // for each image, create an association
  		//   associations.push_back(association);
      }


		  // // create template from associations, role, template
		  // janus_template template1;
		  // janus_template_role role;
		  // int res = janus_create_template(associations, role, template1);
      //
      // for (int i = 0; i < size; i ++) {
      //   janus_media media = associations.media;
      //   janus_free_media(media);
      // }
      //
      // // ====== search ==========
      //
      // // create janus_gallery from gallery path
      // ifstream gallery_stream(gallery_path.c_str(), ios::in | ios::binary);
      // janus_gallery gallery = NULL;
      // janus_deserialize_gallery(gallery, gallery_stream);
      //
      // vector<janus_template_id> return_template_ids;
      // vector<double> similarities;
      // janus_search(template1, gallery, 50, return_template_ids, similarities)
      // janus_delete_gallery(gallery);
      //
      // json::wvalue result;
      // for (int i = 0; i < return_template_ids.size(); i++) {
      //   result[i]["template_id"] = return_template_ids[i];
      //   result[i]["similarity"] = similarities[i];
      // }

      json::wvalue result;
      result[0]["template_id"] = 1;
      result[0]["similarity"] = 0.1;
      result[1]["template_id"] = 2;
      result[1]["similarity"] = 0.2;
      return response(result);

    });

		CROW_ROUTE(app, "/finalize")
    ([]() {
        if (janus_finalize() == 0) {
					return "successful";
				} else {
					return "failed";
				}
    });

    app.port(18080).run();
}

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <csignal>
#include "crow_all.h"
#include "iarpa_janus_io.h"
#include "iarpa_janus.h"
#include "janus_debug.h"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

// to compile
// c++ -Wall -std=c++1y -pedantic -Wextra c_example.cpp -I include -Llib -lboost_system (-lglaive) -o c_example.out
using namespace std;
using namespace crow;
using namespace cv;

void signalHandler( int signum ) {
   cout << "Finalize Janus" << endl;
   janus_finalize();
   exit(signum);

}

int main()
{
    crow::SimpleApp app;

    // clear upload directory
    // TODO change to the right directory at deployment
    std::system("exec rm -r /nfs/div2/jchen/face-search/uploads/*");

    // initialize janus
    cout << "Initialize Janus" << endl;
    janus_initialize("/nfs/isicvlnas01/users/xpeng/projects/Janus/release/janus-isi-sdk-feb/", "/tmp/", "", 0);

    // finalize janus when program abort
    signal(SIGINT, signalHandler);

    CROW_ROUTE(app, "/autodetect")
    .methods("POST"_method)
    ([](const request& req) {
      auto body = json::load(req.body);
      if (!body)
        return response(400);

      string image_path = json::dump(body["image_path"]);
      // remove double quotes from string
      image_path.erase(remove(image_path.begin(), image_path.end(), '\"' ), image_path.end());

      janus_media media;
      janus_load_media(image_path, media);

      cout << "========== CServer: detect bounding box ============" << endl;
      vector<janus_track> tracks;
      janus_detect(media, 50, tracks);
      janus_free_media(media);

      json::wvalue result;
      janus_attributes attributes;
      for (int i = 0; i < tracks.size(); i++) {
         attributes = tracks[i].track[0];
         result[i]["face_x"] = attributes.face_x;
         result[i]["face_y"] = attributes.face_y;
         result[i]["face_width"] = attributes.face_width;
         result[i]["face_height"] = attributes.face_height;
      }

      return response(result);
    });

    CROW_ROUTE(app, "/debug")
    .methods("POST"_method)
    ([](const request& req) {
      auto body = json::load(req.body);
      if (!body)
        return response(400);

      string image_path = json::dump(body["image_path"]);
      // remove double quotes from string
      image_path.erase(remove(image_path.begin(), image_path.end(), '\"' ), image_path.end());

      janus_media media;
      janus_load_media(image_path, media);

      double face_x = body["face_x"].d();
      double face_y = body["face_y"].d();
      double face_width = body["face_width"].d();
      double face_height = body["face_height"].d();

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

      janus_track track;
      track.track.push_back(attributes);

      // create janus_association from janus_media, janus_track
      janus_association association;
      association.media = media;
      association.metadata = track;
      janus_template_role role;

      cv::Mat cropped, rend_fr, rend_hp, rend_fp, aligned;
      float yaw, confidence;
      std::vector<cv::Point2f> landmarks;
      unsigned int landmark_dur, pose_dur, render_dur, align_dur, featex_dur, featex_batch_dur;

      janus_debug(association, role, cropped, rend_fr, rend_hp, rend_fp, aligned, yaw, landmarks, confidence,
        landmark_dur, pose_dur, render_dur, align_dur, featex_dur, featex_batch_dur);

      janus_free_media(media);

      json::wvalue result;
      result["yaw"] = yaw;
      result["confidence"] = confidence;
      for (int i = 0; i < landmarks.size(); i++){
        result["landmarks"][i]["x"] = landmarks[i].x;
        result["landmarks"][i]["y"] = landmarks[i].y;
      }
      result["landmark_dur"] = landmark_dur;
      result["pose_dur"] = pose_dur;
      result["render_dur"] = render_dur;
      result["align_dur"] = align_dur;
      result["featex_dur"] = featex_dur;
      result["featex_batch_dur"] = featex_batch_dur;

      string path;

      if (!rend_fr.empty()) {
        path = image_path + "_rend_fr.jpg";
        imwrite(path, rend_fr);
        // pass only file name
        boost::filesystem::path p1(path);
        result["rend_fr"] = p1.filename().string();
      }

      if (!cropped.empty()) {
        path = image_path + "_cropped.jpg";
        imwrite(path, cropped);
        boost::filesystem::path p2(path);
        result["cropped"] = p2.filename().string();
      }

      if (!rend_hp.empty()) {
        path = image_path + "_rend_hp.jpg";
        imwrite(path, rend_hp);
        boost::filesystem::path p3(path);
        result["rend_hp"] = p3.filename().string();
      }

      if (!rend_fp.empty()) {
        path = image_path + "_rend_fp.jpg";
        imwrite(path, rend_fp);
        boost::filesystem::path p4(path);
        result["rend_fp"] = p4.filename().string();
      }

      if (!aligned.empty()) {
        path = image_path + "_aligned.jpg";
        imwrite(path, aligned);
        boost::filesystem::path p5(path);
        result["aligned"] = p5.filename().string();
      }

      return response(result);
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

      cout << "========== CServer: Process inputs ==============" << endl;

      // get settings, which is first object in array
      int maxResults = arr[0]["maxResults"].i();
      if (maxResults > 50 || maxResults < 0) {
        maxResults = 20;
      }

      int image_count = arr.size() - 1;
      string image_paths[image_count];
      vector<janus_association> associations;
      for (int i = 1; i < image_count+1; i ++) {
        auto img = arr[i];
        string image_path = json::dump(img["image_path"]);
        // remove double quotes from string
        image_path.erase(remove(image_path.begin(), image_path.end(), '\"' ), image_path.end());
        image_paths[i-1] = image_path;

        // create media from image path
        janus_media media;
        janus_load_media(image_path, media);

        // create janus_track from janus_attributes
  		  janus_track track;

        // if has attributes
        if (img.has("face_x")) {
          cout << "has bounding box" << endl;
          double face_x = img["face_x"].d();
          double face_y = img["face_y"].d();
          double face_width = img["face_width"].d();
          double face_height = img["face_height"].d();

          cout << image_path << ":" << to_string(face_x) << "," << to_string(face_y) << "," << to_string(face_width) << "," << to_string(face_height) << endl;

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

  		    track.track.push_back(attributes);


          // create janus_association from janus_media, janus_track
    		  janus_association association;
    		  association.media = media;
    		  association.metadata = track;

          // for each image, create an association
    		  associations.push_back(association);
        } else { // no attributes

          cout << "has no bounding box" << endl;
          vector<janus_track> tracks;
          janus_detect(media, 50, tracks);

          janus_association association;
    		  association.media = media;
    		  association.metadata = tracks[0];
          associations.push_back(association);
        }
      }

      cout << "========== CServer: Creating Template ==============" << endl;

		  // create template from associations, role, template
		  janus_template template1;
		  janus_template_role role;
      Mat out_cropped[image_count];
    	Mat out_rend_fr[image_count];
    	Mat out_rend_hp[image_count];
    	Mat out_rend_fp[image_count];
    	Mat out_aligned[image_count];
    	float out_yaw[image_count];
    	vector<Point2f> out_landmarks[image_count];
    	float out_confidence[image_count];
      unsigned int landmark_dur[image_count];
      unsigned int pose_dur[image_count];
      unsigned int render_dur[image_count];
      unsigned int align_dur[image_count];
      unsigned int featex_dur[image_count];
      unsigned int featex_batch_dur[image_count];

		  // int ct_res = janus_create_template(associations, role, template1);
      int ct_res = janus_create_template_debug(associations, role, template1,
        out_cropped, out_rend_fr, out_rend_hp, out_rend_fp, out_aligned, out_yaw, out_landmarks, out_confidence,
        landmark_dur, pose_dur, render_dur, align_dur, featex_dur, featex_batch_dur);

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
      int s_res = janus_search(template1, gallery, maxResults, return_template_ids, similarities);
      janus_delete_gallery(gallery);
      janus_delete_template(template1);

      if (s_res != 0)
        return response("janus_search failed");



      json::wvalue result;
      for (int i = 0; i < return_template_ids.size(); i++) {
        result["results"][i]["template_id"] = return_template_ids[i];
        result["results"][i]["similarity"] = similarities[i];
      }

      for (int i = 0; i < image_count; i++) {
        string image_path = image_paths[i];
        boost::filesystem::path p(image_path);
        string filename = p.filename().string();

        result[filename]["yaw"] = out_yaw[i];
        result[filename]["confidence"] = out_confidence[i];
        for (int j = 0; j < out_landmarks[i].size(); j++){
          result[filename]["landmarks"][j]["x"] = out_landmarks[i][j].x;
          result[filename]["landmarks"][j]["y"] = out_landmarks[i][j].y;
        }
        result[filename]["landmark_dur"] = landmark_dur[i];
        result[filename]["pose_dur"] = pose_dur[i];
        result[filename]["render_dur"] = render_dur[i];
        result[filename]["align_dur"] = align_dur[i];
        result[filename]["featex_dur"] = featex_dur[i];
        result[filename]["featex_batch_dur"] = featex_batch_dur[i];

        string path;

        if (!out_rend_fr[i].empty()) {
          path = image_path + "_rend_fr.jpg";
          imwrite(path, out_rend_fr[i]);
          // pass only file name
          boost::filesystem::path p1(path);
          result[filename]["rend_fr"] = p1.filename().string();
        }

        if (!out_cropped[i].empty()) {
          path = image_path + "_cropped.jpg";
          imwrite(path, out_cropped[i]);
          boost::filesystem::path p2(path);
          result[filename]["cropped"] = p2.filename().string();
        }

        if (!out_rend_hp[i].empty()) {
          path = image_path + "_rend_hp.jpg";
          imwrite(path, out_rend_hp[i]);
          boost::filesystem::path p3(path);
          result[filename]["rend_hp"] = p3.filename().string();
        }

        if (!out_rend_fp[i].empty()) {
          path = image_path + "_rend_fp.jpg";
          imwrite(path, out_rend_fp[i]);
          boost::filesystem::path p4(path);
          result[filename]["rend_fp"] = p4.filename().string();
        }

        if (!out_aligned[i].empty()) {
          path = image_path + "_aligned.jpg";
          imwrite(path, out_aligned[i]);
          boost::filesystem::path p5(path);
          result[filename]["aligned"] = p5.filename().string();
        }
      }

      return response(result);

    });

    app.port(8081).run();
}

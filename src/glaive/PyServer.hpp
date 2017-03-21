#ifndef PYSERVER_HPP
#define PYSERVER_HPP

#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <map>
#include <pthread.h>

#include "iarpa_janus.h"

class PyServer
{
public:
  PyServer() {};

  janus_error spawnPyWorker(std::map<std::string, std::string> py_params);
  int getLock();
  void spawnPyWorkerViaFork();
  void initializeServer();

  int getSocket();

  static ssize_t readMat(int socket, cv::Mat &m);
  static ssize_t writeVideoBuf( int socket, const int frames, const int width, const int height, uint8_t *video_buf );
  static ssize_t writeBox( int socket, const int minx, const int maxx, const int miny, const int maxy );
  static ssize_t writeData(int socket, void *buf, size_t numBytes);
  static ssize_t readData(int socket, void *buf, size_t numBytes);
  static ssize_t writeVector(int socket, std::vector<float> v);
  static ssize_t readVector(int socket, std::vector<float> &v);

private:
  bool try_socket();

  std::map<std::string, std::string> py_params;

  const std::string PY_SOCKET_PATH = "/tmp/isi-janus-py-worker.socket";
  const std::string LOCKFILE = "/tmp/isi-janus-py-worker.lockfile";

  int batchSize = 64;

  int g_lockfile_fd;

  int serverDone  = 0;
  int clientThreadCount = 0;

  const suseconds_t MAX_WAIT_SEC = 0; // 
  const suseconds_t MAX_WAIT_USEC = 999999; // ~1 seconds


  pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
  // Condition for signaling server thread has finished initializing
  pthread_cond_t server_ready_cond = PTHREAD_COND_INITIALIZER;
  int serverThreadReady = 0;

  // Condition for signaling ready to accept more input
  pthread_cond_t input_ready_cond = PTHREAD_COND_INITIALIZER;

  // Conition for signaling some input is ready for procesing
  pthread_cond_t process_cond = PTHREAD_COND_INITIALIZER;

  // Condition for signaling cnn output is ready and for signaling output has been ready
  pthread_cond_t result_ready_cond = PTHREAD_COND_INITIALIZER;

  // Conition for signaling one output has been read
  pthread_cond_t output_read_cond = PTHREAD_COND_INITIALIZER;

  // input queue
  std::vector< cv::Mat* > input_image_list;

  // ouput queue
  std::vector<std::vector<float>> feature_vector_list;

  // Can add to input queue, if true
  int inputReady = 1;

  // Can read from output queue, if true
  int outputReady = 0;

  // Length of output queue
  int pyResultCount = 0;


};


#endif

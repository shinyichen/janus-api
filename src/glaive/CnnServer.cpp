#include "CnnServer.hpp"

#include <iostream>
#include <stdio.h>
#include <signal.h>
#include <thread>
#include <poll.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <chrono>


#include <boost/filesystem.hpp>


janus_error CnnServer::spawnCnnWorker(std::string p_sdk_path, const int p_gpu_dev)
{
  this->sdk_path = p_sdk_path;
  this->gpu_dev = p_gpu_dev;

  // Setup lock/socket vars using gpu_dev (to allow multiple processes to use different GPU's)
  CNN_SOCKET_PATH = CNN_SOCKET_PATH_PREFIX + std::to_string(gpu_dev) + ".socket";
  LOCKFILE = LOCKFILE_PREFIX + std::to_string(gpu_dev) + ".lockfile";


  int haveLock = getLock();

  if (haveLock) {
    // Need to check if server is already initialized
    if (!try_socket()) {
      spawnCnnWorkerViaFork();
    }
  } else {
    // wait for server to be initialized
    while(1) {
      if (try_socket())
	break;
      else
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }
  close(g_lockfile_fd);

  keepCnnWorkerAlive();

  return JANUS_SUCCESS;
}

bool CnnServer::try_socket()
{
  struct sockaddr_un addr;
  int socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);      /* Create client socket */
  if (socket_fd == -1) return false;

  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, CNN_SOCKET_PATH.c_str(), sizeof(addr.sun_path) - 1);
  int status = connect(socket_fd, (struct sockaddr *) &addr, sizeof(struct sockaddr_un));

  if (status == -1) return false;

  close(socket_fd);

  return true;
}

int CnnServer::getSocket()
{
  struct sockaddr_un addr;
  int socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);      /* Create client socket */
  if (socket_fd == -1) {
    perror("socket");
    exit(1);
  }

  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, CNN_SOCKET_PATH.c_str(), sizeof(addr.sun_path) - 1);
  int status = connect(socket_fd, (struct sockaddr *) &addr, sizeof(struct sockaddr_un));

  if (status == -1) {
    std::cerr << "Failed connect first time. Let's sleep a little and try again. (for socket path: " << addr.sun_path << ")" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    status = connect(socket_fd, (struct sockaddr *) &addr,
		     sizeof(struct sockaddr_un));

    if (status == -1) {
      std::cerr << "Failed connect inside SpawnSingletonServers: " << addr.sun_path << std::endl;
      perror("Socket connect error");
      exit(1);
    }

  }

  return socket_fd;
}

void CnnServer::keepCnnWorkerAlive()
{
  // Create lambda which will run in background thread
  auto keep_alive_lambda = [this] () {
    while (1) {
      std::this_thread::sleep_for(std::chrono::seconds(120)); // TODO: make this + server timeout configurable!!

      int socket_fd = getSocket();

      char command = 'a'; // keep-alive command
      size_t bytes;

      bytes = write(socket_fd, &command, sizeof(char));
      if (bytes == -1) {
	perror("write socket");
      }
      close(socket_fd);
    }
  };

  // Setoff as background thread
  std::thread *backgroundThread = new std::thread(keep_alive_lambda);
}


int CnnServer::getLock()
{
  // Attempt to obtain lock on lockfile; if we get it return 1, else return 0

  int haveLock;
  if (! boost::filesystem::exists(LOCKFILE) ) {
    if ( (g_lockfile_fd = open(LOCKFILE.c_str(), O_CREAT, S_IRWXU)) == -1) {
      std::cerr << "Failed to create file: " << LOCKFILE << std::endl;
      perror("open create");
      return -1;
    }
  }
  // get file descriptor to lockfile
  if ( (g_lockfile_fd = open(LOCKFILE.c_str(), O_WRONLY)) == -1) {
    perror("open");
    return -1;
  }
  int lockOp = LOCK_EX | LOCK_NB; // exclusive lock with non-blocking
  if (flock(g_lockfile_fd, lockOp) == -1) {
    if (errno == EWOULDBLOCK)  {
      haveLock = 0;
    } else {
      return -1;
    }
  } else {
    haveLock = 1;
  }

  return haveLock;
}


void CnnServer::spawnCnnWorkerViaFork()
{
  // Ignore SIGCHLD to avoid creating zombie processes
  signal(SIGCHLD, SIG_IGN);

  if (!fork()) { 
    /* Child */
    initializeServer();
    exit(0);
  }

  sleep(20);
}

void CnnServer::initializeServer()
{
  std::thread serverThread( [this] { serverThreadFunc(); } );
  serverThread.detach();

  struct sockaddr_un addr;
  int sfd, cfd;

  sfd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sfd == -1) {
    perror("socket");
    exit(1);
  }

  if (remove(CNN_SOCKET_PATH.c_str()) == -1 && errno != ENOENT) {
    perror("remove");
    exit(1);
  }

  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, CNN_SOCKET_PATH.c_str(), sizeof(addr.sun_path) - 1);

  if (bind(sfd, (struct sockaddr *) &addr, sizeof(struct sockaddr_un)) == -1) {
    perror("bind");
    exit(1);
  }

  int backlog = 256;
  if (listen(sfd, backlog) == -1) {
    perror("listen");
    exit(1);
  }

  pollfd pfd;
  pfd.fd = sfd;
  pfd.events = POLLIN;

  int sleepCount = 0;
  int timeOutExitServer = 5 * 60; // 5 minutes

  while (!serverDone) {          /* Handle client connections iteratively */
    /* Accept a connection. The connection is returned on a new
       socket, 'cfd'; the listening socket ('sfd') remains open
       and can be used to accept further connections. */

    int pollStatus;

    // check if any clients try to connect
    pollStatus = poll(&pfd, 1, 30000);

    if (pollStatus < 0) {
      perror("poll");
      exit(1);
    } else if (pollStatus == 0) {
      sleepCount += 30;
      if (sleepCount >= timeOutExitServer) {
	serverDone = 1;
      }
    } else {
      sleepCount = 0;
      cfd = accept(sfd, NULL, NULL);
      if (cfd == -1) {
	perror("accept");
	exit(1);
      }

      pthread_mutex_lock(&mtx);
      ++clientThreadCount;
      pthread_mutex_unlock(&mtx);

      std::thread clientThread( [this, cfd] { clientThreadFunc(cfd); } );
      clientThread.detach();
    }
  }
}


void * CnnServer::serverThreadFunc()
{
  // Server thread ready and input queue ready
  {
    pthread_mutex_lock(&mtx);
    serverThreadReady = 1;
    inputReady = 1;
    pthread_cond_broadcast(&server_ready_cond);
    pthread_mutex_unlock(&mtx);
  }

  struct timespec wait;
  wait.tv_sec = 0;
  wait.tv_nsec = 1000 * MAX_WAIT_USEC;

  // Initialize featex here, so we can start accepting clients right away
  // We won't run featex until this line is done anyway (and we have init checks just in case too)
  cnn_featex.initialize(sdk_path, gpu_dev);
  while (! serverDone) {
    int waitForMoreInput = 1;
    int firstImage = 1;
    int timedOut = 0;
    struct timeval firstImageTime, curr;
    suseconds_t delta = 0;

    // Gather a batch of images
    pthread_mutex_lock(&mtx);
    while (waitForMoreInput) {
      int status = pthread_cond_timedwait(&process_cond, &mtx, &wait);

      if (status == ETIMEDOUT) {
	// Timed out waiting. 
	// If there are images, process them
	if (input_image_list.size() > 0) {
	  waitForMoreInput = 0;
	  inputReady = 0;
	  timedOut = 1;
	} else {
	  wait.tv_nsec = 1000 * MAX_WAIT_USEC;
	}
      } else {
	if (input_image_list.size() > 0) {
	  if (firstImage) {
	    // For the firstImage make a note of the time
	    firstImage = 0;
	    gettimeofday(&firstImageTime, NULL);
	    wait.tv_nsec = 1000 * MAX_WAIT_USEC;
	  } else {
	    if (input_image_list.size() >= batchSize) {
	      waitForMoreInput = 0;
	      inputReady = 0;
	    } else {
	      // wait some more
	      gettimeofday(&curr, NULL);
	      delta = 1000000 * (curr.tv_sec - firstImageTime.tv_sec) 
		+ (curr.tv_usec - firstImageTime.tv_usec);
	      if (delta >= MAX_WAIT_USEC) {
		waitForMoreInput = 0;
		inputReady = 0;
		timedOut = 1;
	      } else {
		wait.tv_nsec = 1000 * (MAX_WAIT_USEC - delta);
	      }
	    }
	  }
	} else {
	  wait.tv_nsec = 1000 * MAX_WAIT_USEC;
	}
      }
    }
    pthread_mutex_unlock(&mtx);

    // Run CNN feature extractor
    if (input_image_list.size() == 0) {
      std::cout << "ERROR: shouldn't get here with input image list size of 0" << std::endl;
    }


    auto start_time = std::chrono::steady_clock::now();
    feature_vector_list = cnn_featex.extract_batch(input_image_list);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
    std::cout << "Raw featex (w/o synchronization cost) on " << input_image_list.size() << " images took: " << duration.count() << " ms" << std::endl;

    // Signal Cnn features are ready
    {
      pthread_mutex_lock(&mtx);
      outputReady = 1;
      cnnResultCount = feature_vector_list.size();
      pthread_cond_broadcast(&result_ready_cond);
      pthread_mutex_unlock(&mtx);
    }


    // Wait until all the output are read before continuing
    {
      pthread_mutex_lock(&mtx);
      while (cnnResultCount > 0) {
	pthread_cond_wait(&output_read_cond, &mtx);
      }
      outputReady = 0;
      feature_vector_list.clear();
      pthread_mutex_unlock(&mtx);
      inputReady = 1;
      input_image_list.clear();
      pthread_cond_broadcast(&input_ready_cond);
    }
  }

}

void *CnnServer::clientThreadFunc(int socket) {

   // Wait for server thread ready
  {
    pthread_mutex_lock(&mtx);
    while (serverThreadReady ==0 ) {
      pthread_cond_wait(&server_ready_cond, &mtx);
    }
    pthread_mutex_unlock(&mtx);
  }

  int imageIndex;
  char command;

  int numRead = read(socket, &command, sizeof(command));

  if (numRead == -1) {
    perror("read");
  }

  if (command == 'p') { // p = accept 1 image
    cv::Mat image;
    std::vector<float> feature_vector;
      
    // Get image from client
    readMat(socket, image);

    // Waite for input list ready
    {
      pthread_mutex_lock(&mtx);

      while (!inputReady) {
	pthread_cond_wait(&input_ready_cond, &mtx);
      }
      
      imageIndex = input_image_list.size();
      input_image_list.push_back(&image);
      if (input_image_list.size() >= batchSize) {
	inputReady = 0;
      } 

      pthread_mutex_unlock(&mtx);
    }

    // Signal processing thread an input is ready
    pthread_cond_signal(&process_cond);

    // Wait for result
    {
      pthread_mutex_lock(&mtx);

      while (!outputReady) {
	pthread_cond_wait(&result_ready_cond, &mtx);
      }
      
      if (imageIndex >= feature_vector_list.size()) {
	perror("imageIndex >= feature_vector_list.size()");
	exit(1);
      }

      feature_vector = feature_vector_list[imageIndex];
      --cnnResultCount;
      pthread_cond_signal(&output_read_cond);

      if (cnnResultCount < 0) {
	perror("cnnResultCount < 0");
	exit(1);
      }
      pthread_mutex_unlock(&mtx);
    }

    // send feature back to client
    int status = 0;
    writeData(socket, &status, sizeof(status));
      
    writeVector(socket, feature_vector);
  } else if (command == 'P') { // P (capital) = accept multiple images

    // First read in how many images we are dealing with
    int num_images;
    int numRead = read(socket, &num_images, sizeof(num_images));

    if (numRead == -1) {
      perror("read");
    }

    std::vector<cv::Mat> images;
    for (int i = 0; i < num_images; ++i) {  
      cv::Mat image;
      readMat(socket, image);
      images.push_back(image);
    }


    // Waite for input list ready
    {
      pthread_mutex_lock(&mtx);

      while (!inputReady) {
	pthread_cond_wait(&input_ready_cond, &mtx);
      }
      
      imageIndex = input_image_list.size();

      for (int i = 0; i < num_images; ++i)
	input_image_list.push_back(&(images[i]));

      if (input_image_list.size() >= batchSize) {
	inputReady = 0;
      } 

      pthread_mutex_unlock(&mtx);
    }

    // Signal processing thread an input is ready
    pthread_cond_signal(&process_cond);


    auto start_time = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> feature_vectors;
    // Wait for result
    {
      pthread_mutex_lock(&mtx);

      while (!outputReady) {
	pthread_cond_wait(&result_ready_cond, &mtx);
      }
      
      if (imageIndex >= feature_vector_list.size()) {
	perror("imageIndex >= feature_vector_list.size()");
	exit(1);
      }

      if (imageIndex+num_images > feature_vector_list.size()) {
	perror("imageIndex+num_images >= feature_vector_list.size()");
	exit(1);
      }


      std::cout << "Starting to communicate feature vectors back out" << std::endl;
      start_time = std::chrono::steady_clock::now();

      for (int i = 0; i < num_images; ++i) {
	feature_vectors.push_back(feature_vector_list[imageIndex+i]);
	--cnnResultCount;
      }

      pthread_cond_signal(&output_read_cond);

      if (cnnResultCount < 0) {
	perror("cnnResultCount < 0");
	exit(1);
      }
      pthread_mutex_unlock(&mtx);
    }

    // send feature back to client
    int status = 0;
    writeData(socket, &status, sizeof(status));
      
    std::cout << "Writing vectors out" << std::endl;
    for (int i = 0; i < num_images; ++i) {
      writeVector(socket, feature_vectors[i]);
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
    std::cout << "Done writing batch of vectors" << std::endl;

    std::cout << "Broadcast + communication cost for featex took: " << duration.count()<< " ms" << std::endl;


  } else if (command == 'q') {
    std::cout << "Quit command" << std::endl;
  } else if (command == 'a') {
    std::cout << "Keep Alive command" << std::endl;
  }
  else {
    std::cout << "Unknown command: [" << command << "]" << std::endl;
  }

  {
    pthread_mutex_lock(&mtx);
    --clientThreadCount;
    pthread_mutex_unlock(&mtx);
  }

  close(socket);
}


ssize_t CnnServer::readMat(int socket, cv::Mat &m) {

  int cols, rows, type;
  size_t elemSize, step;

  ssize_t bytes = 0;
  
  bytes += read(socket, &cols, sizeof(int));
  bytes += read(socket, &rows, sizeof(int));
  bytes += read(socket, &elemSize, sizeof(size_t));
  bytes += read(socket, &type, sizeof(int));
  bytes += read(socket, &step, sizeof(size_t));

  //size_t len = cols * rows * elemSize;
  size_t len = rows * step;

  std::cout << "About to malloc [" << len << "] bytes" << std::endl;
  uchar *data = (uchar *) malloc(len);
  bytes += readData(socket, data, len);

  cv::Mat copy = cv::Mat(rows, cols, type, data, step);
  m = copy.clone();
  free(data);

  return bytes;
}

ssize_t CnnServer::writeMat(int socket, const cv::Mat m) {
  ssize_t bytes = 0;
  
  size_t elemSize = m.elemSize();
  int mtype = m.type();
  size_t step = m.step;
  bytes += write(socket, &(m.cols), sizeof(int));
  bytes += write(socket, &(m.rows), sizeof(int));
  bytes += write(socket, &(elemSize), sizeof(size_t));
  bytes += write(socket, &(mtype), sizeof(int));
  bytes += write(socket, &(step), sizeof(size_t));

  //size_t len = m.cols * m.rows * m.elemSize();
  size_t len = m.rows * step;
  bytes += writeData(socket, m.data, len);
  return bytes;
}


ssize_t CnnServer::writeData(int socket, void *buf, size_t numBytes) {
  size_t offset = 0;
  ssize_t b = 1;
  //printf("Start writing %lu\n", numBytes);
  while (offset < numBytes && b > 0) {
    b = write(socket, ((char *) buf) + offset, numBytes-offset);
    if (b < 0) {
      printf("Socket Error");
    }
    offset += b;
  }
  //printf("Done writing %lu\n", numBytes);
  if (b < 0) {
    return b;
  } else {
    return offset;
  }
}

ssize_t CnnServer::readData(int socket, void *buf, size_t numBytes) {
  size_t offset = 0;
  ssize_t b = 1;

  //printf("Start reading %lu\n", numBytes);
  while (offset < numBytes && b > 0) {
    b = read(socket, ((char *) buf) + offset, numBytes-offset);
    if (b < 0) {
      printf("Socket Error");
    }
    offset += b;
  }

  //printf("Done reading %lu\n", numBytes);
  if (b < 0) {
    return b;
  } else {
    return offset;
  }
}

ssize_t CnnServer::writeVector(int socket, std::vector<float> v) {
  size_t size = v.size();
  float * ptr = v.data();
  ssize_t bytes;
  bytes = write(socket, &size, sizeof(size));
  bytes += writeData(socket, ptr, v.size() * sizeof(float));
  return bytes;
}


ssize_t CnnServer::readVector(int socket, std::vector<float> &v) {
  size_t size;
  ssize_t bytes = read(socket, &size, sizeof(size_t));

  v.resize(size);
  float *buf = v.data();
  bytes += readData(socket, buf, size*sizeof(float));
  return bytes;

}

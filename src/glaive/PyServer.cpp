#include "PyServer.hpp"

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
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <map>

#include <boost/filesystem.hpp>
#include <opencv2/imgproc.hpp>


janus_error PyServer::spawnPyWorker(std::map<std::string, std::string> py_params)
{
  this->py_params = py_params;

  int haveLock = getLock();

  if (haveLock) {
    spawnPyWorkerViaFork();
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

  return JANUS_SUCCESS;
}

bool PyServer::try_socket()
{
  struct sockaddr_un addr;
  int socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);      /* Create client socket */
  if (socket_fd == -1) return false;

  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, PY_SOCKET_PATH.c_str(), sizeof(addr.sun_path) - 1);
  int status = connect(socket_fd, (struct sockaddr *) &addr, sizeof(struct sockaddr_un));

  if (status == -1) return false;

  close(socket_fd);

  return true;
}

int PyServer::getSocket()
{
  struct sockaddr_un addr;
  int socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);      /* Create client socket */
  if (socket_fd == -1) {
    perror("socket");
    exit(1);
  }

  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, PY_SOCKET_PATH.c_str(), sizeof(addr.sun_path) - 1);
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


int PyServer::getLock()
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


void PyServer::spawnPyWorkerViaFork()
{
  // Ignore SIGCHLD to avoid creating zombie processes
  signal(SIGCHLD, SIG_IGN);

  if (!fork()) { 
    /* Child */
    initializeServer();
    exit(0);
  }

  sleep(10);
}

void PyServer::initializeServer()
{
  // Start the python server
  std::map<std::string, std::string>::iterator py_bin    = this->py_params.find("py_bin");
  std::map<std::string, std::string>::iterator py_server = this->py_params.find("py_server");

  if ( py_bin != this->py_params.end() && py_server != this->py_params.end()) {
    execl(py_bin->second.c_str(), py_bin->second.c_str(), py_server->second.c_str(), (char*)0);
  }
  else {
    std::cerr << "Initialize server failed." << std::endl;
  }  
    
}




ssize_t PyServer::readMat(int socket, cv::Mat &m) {

  int cols, rows, type;
  size_t elemSize, step;

  ssize_t bytes = 0;
  
  bytes += read(socket, &cols, sizeof(int));
  bytes += read(socket, &rows, sizeof(int));
  bytes += read(socket, &elemSize, sizeof(size_t));
  bytes += read(socket, &type, sizeof(int));
  bytes += read(socket, &step, sizeof(size_t));

  std::cout << cols << " ";
  std::cout << rows << " ";
  std::cout << elemSize << " ";
  std::cout << type << " ";
  std::cout << step << std::endl;

  //size_t len = cols * rows * elemSize;
  size_t len = rows * step;

  std::cout << "cols * rows * elemSize =" << cols * rows * elemSize << std::endl;
  std::cout << "m.rows * step =" << len << std::endl;

  uchar *data = (uchar *) malloc(len);
  bytes += readData(socket, data, len);

  cv::Mat copy = cv::Mat(rows, cols, type, data, step);
  m = copy.clone();
  free(data);

  std::cout << "x " << m.cols << " ";
  std::cout << m.rows << " ";
  std::cout << m.elemSize() << " ";
  std::cout << m.type() << " ";
  std::cout << m.step << std::endl;

  return bytes;
}

ssize_t PyServer::writeVideoBuf(int socket, const int frames, const int width, const int height, uint8_t *video_buf ) {
  ssize_t bytes = 0;
  
  bytes += write(socket, &frames, sizeof(int));
  bytes += write(socket, &width, sizeof(int));
  bytes += write(socket, &height, sizeof(int));

  size_t len = frames*width*height;

  bytes += writeData(socket, video_buf, sizeof(uint8_t)*len);

  return bytes;
}

ssize_t PyServer::writeBox( int socket, const int minx, const int maxx, const int miny, const int maxy )
{
  ssize_t bytes = 0;

  bytes += write(socket, &minx, sizeof(int));
  bytes += write(socket, &maxx, sizeof(int));
  bytes += write(socket, &miny, sizeof(int));
  bytes += write(socket, &maxy, sizeof(int));

  return bytes;
}

ssize_t PyServer::writeData(int socket, void *buf, size_t numBytes) {
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

ssize_t PyServer::readData(int socket, void *buf, size_t numBytes) {
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

ssize_t PyServer::writeVector(int socket, std::vector<float> v) {
  size_t size = v.size();
  float * ptr = v.data();
  ssize_t bytes;
  bytes = write(socket, &size, sizeof(size));
  bytes += writeData(socket, ptr, v.size() * sizeof(float));
  return bytes;
}


ssize_t PyServer::readVector(int socket, std::vector<float> &v) {
  size_t size;
  ssize_t bytes = read(socket, &size, sizeof(size_t));

  v.resize(size);
  float *buf = v.data();
  bytes += readData(socket, buf, size*sizeof(float));
  return bytes;

}

#include <Python.h>
#include <numpy/arrayobject.h>

#include <iostream>

#include "python_wrappers.h"


janus_error init_lm2pose(std::string poseConfigFile);
janus_error init_render();

PyObject *poseModule = NULL;
PyObject *renderModule = NULL;
PyObject *pGetPose = NULL;
PyObject *pRenderFunc = NULL;

PyThreadState *_threadstate_save;

janus_error
pythonInitialize(std::string pythonPath, std::string poseConfigFile) {

  Py_Initialize();
  PyEval_InitThreads();

  // Add to python path
  std::string path = std::string(Py_GetPath());
  path = path + ":" + pythonPath;
  char newpath[path.length()+1];
  std::strcpy(newpath, path.c_str());
  PySys_SetPath(newpath);

  std::cout << "Initializing LM2Pose" << std::endl;
  janus_error status = init_lm2pose(poseConfigFile);
  if (status != JANUS_SUCCESS) return status;

  std::cout  << "Initializing Renderer" << std::endl;
  status = init_render();
  if (status != JANUS_SUCCESS) return status;

  _threadstate_save = PyEval_SaveThread();

  return JANUS_SUCCESS;
}

janus_error init_render() {
  PyObject *rName;
  rName = PyString_FromString("renderer_api");

  renderModule = PyImport_Import(rName);
  Py_DECREF(rName);

  if (renderModule == NULL) {
    PyErr_Print();
    fprintf(stderr, "Failed to load render.py\n");
    return JANUS_UNKNOWN_ERROR;
  }

  // Now let's try to init
  std::string initFuncStr = "init";
  PyObject *renderInit;
  renderInit = PyObject_GetAttrString(renderModule, initFuncStr.c_str());

  if (!renderInit || !PyCallable_Check(renderInit)) {
    if (PyErr_Occurred())
      PyErr_Print();
    fprintf(stderr, "Cannot find function \"%s\"\n", initFuncStr.c_str());
    return JANUS_UNKNOWN_ERROR;
  }

  PyObject *result = PyObject_CallObject(renderInit, NULL);
  if (result == NULL) {
    if (PyErr_Occurred()) {
      PyErr_Print();
      return JANUS_UNKNOWN_ERROR;
    }
  }

  pRenderFunc = PyObject_GetAttrString(renderModule, "render");
  if (!(pRenderFunc && PyCallable_Check(pRenderFunc))) {
    if (PyErr_Occurred())
      PyErr_Print();
    fprintf(stderr, "Cannot find python function render\n");
    Py_XDECREF(pRenderFunc);
    return JANUS_UNKNOWN_ERROR;
  }

  return JANUS_SUCCESS;
}

janus_error init_lm2pose(std::string poseConfigFile) {
  std::string initFuncStr = "init";
  PyObject *pName;
  pName = PyString_FromString("lm2pose");
  /* Error checking of pName left out */

  poseModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (poseModule != NULL) {
    PyObject *poseInit, *pcfile;
    poseInit = PyObject_GetAttrString(poseModule, initFuncStr.c_str());
    if (poseInit && PyCallable_Check(poseInit)) {
      pcfile = PyString_FromString(poseConfigFile.c_str());
      PyObject *pArgs = PyTuple_New(1);
      PyTuple_SetItem(pArgs, 0, pcfile);

      PyObject *result = PyObject_CallObject(poseInit, pArgs);
      if (result == NULL) {
        if (PyErr_Occurred()) {
          PyErr_Print();
          return JANUS_UNKNOWN_ERROR;
        }
      }
      Py_XDECREF(result);
      Py_DECREF(poseInit);
      Py_DECREF(pArgs);
    } else {
      if (PyErr_Occurred())
        PyErr_Print();
      fprintf(stderr, "Cannot find function \"%s\"\n", initFuncStr.c_str());
      return JANUS_UNKNOWN_ERROR;
    }
  } else {
    PyErr_Print();
    fprintf(stderr, "Failed to load lm2pose.py\n");
    return JANUS_UNKNOWN_ERROR;
  }

  // import numpy array module                                                                                                               
  import_array1(JANUS_UNKNOWN_ERROR);

  pGetPose = PyObject_GetAttrString(poseModule, "getpose_no_init");
  if (!(pGetPose && PyCallable_Check(pGetPose))) {
    if (PyErr_Occurred())
      PyErr_Print();
    fprintf(stderr, "Cannot find python function getpose_no_init\n");
    Py_XDECREF(pGetPose);
    return JANUS_UNKNOWN_ERROR;
  }

  return JANUS_SUCCESS;
}

janus_error pythonDoRendering(cv::Mat image_in, std::vector<cv::Point2f> lmrks_in, cv::Mat &rend_fr_out, cv::Mat &rend_hp_out, cv::Mat &rend_fp_out, float &yaw_out) {

  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  // Construct input args
  PyObject *pyImageWrapper;
  npy_intp dim[3] = {(long int)image_in.rows, (long int)image_in.cols, image_in.channels()};
  pyImageWrapper = PyArray_SimpleNewFromData(3, dim, NPY_UINT8, image_in.data);


  PyObject *lmValue, *pValue;
  lmValue = PyTuple_New(2 * lmrks_in.size());
  int j=0;
  for (size_t i=0; i<lmrks_in.size(); ++i, j += 2) {
    pValue = PyFloat_FromDouble(double(lmrks_in[i].x));
    PyTuple_SetItem(lmValue, j, pValue);
    pValue = PyFloat_FromDouble(double(lmrks_in[i].y));
    PyTuple_SetItem(lmValue, j+1, pValue);
  }


  PyObject *pArgs;
  pArgs = PyTuple_New(2);
  PyTuple_SetItem(pArgs, 0, pyImageWrapper);
  PyTuple_SetItem(pArgs, 1, lmValue);


  PyObject *pTuple;
  pTuple = PyObject_CallObject(pRenderFunc, pArgs);
  if ((pTuple == NULL) || !PyTuple_Check(pTuple))  {
    PyErr_Print();
    std::cout << "Error: call to render() failed" << std::endl;
    Py_DECREF(pArgs);
    PyGILState_Release(gstate);
    return JANUS_UNKNOWN_ERROR;
  }

  PyObject *pNumPoses = (PyObject *) PyTuple_GetItem(pTuple, 0);
  PyObject *yaw;

  int numPoses = (int)PyInt_AsLong(pNumPoses);
  if (numPoses == 2) {
    PyArrayObject *p_hp_ImageArray = (PyArrayObject *) PyTuple_GetItem(pTuple, 1);
    PyArrayObject *p_fp_ImageArray = (PyArrayObject *) PyTuple_GetItem(pTuple, 2);
    yaw =  (PyObject*) PyTuple_GetItem(pTuple, 3);
    yaw_out = (float)PyFloat_AsDouble(yaw);

    // (1) Handle half-profile
    int nbytes = PyArray_NBYTES(p_hp_ImageArray);
    uchar *data = (uchar *) malloc(nbytes);
    memcpy(data, PyArray_DATA(p_hp_ImageArray), nbytes);
    cv::Mat copy = cv::Mat(PyArray_DIMS(p_hp_ImageArray)[0], PyArray_DIMS(p_hp_ImageArray)[1], image_in.type(), data);
    rend_hp_out = copy.clone();
    free(data);

    // (2) Handle full-profile
    nbytes = PyArray_NBYTES(p_fp_ImageArray);
    data = (uchar *) malloc(nbytes);
    memcpy(data, PyArray_DATA(p_fp_ImageArray), nbytes);
    copy = cv::Mat(PyArray_DIMS(p_fp_ImageArray)[0], PyArray_DIMS(p_fp_ImageArray)[1], image_in.type(), data);
    rend_fp_out = copy.clone();
    free(data);

  } else if (numPoses == 3) {
    PyArrayObject *p_fr_ImageArray = (PyArrayObject *) PyTuple_GetItem(pTuple, 1);
    PyArrayObject *p_hp_ImageArray = (PyArrayObject *) PyTuple_GetItem(pTuple, 2);
    PyArrayObject *p_fp_ImageArray = (PyArrayObject *) PyTuple_GetItem(pTuple, 3);
    yaw =  (PyObject*) PyTuple_GetItem(pTuple, 4);

    yaw_out = (float)PyFloat_AsDouble(yaw);

    // (1) Handle half-profile
    int nbytes = PyArray_NBYTES(p_hp_ImageArray);
    uchar *data = (uchar *) malloc(nbytes);
    memcpy(data, PyArray_DATA(p_hp_ImageArray), nbytes);
    cv::Mat copy = cv::Mat(PyArray_DIMS(p_hp_ImageArray)[0], PyArray_DIMS(p_hp_ImageArray)[1], image_in.type(), data);
    rend_hp_out = copy.clone();
    free(data);

    // (2) Handle full-profile
    nbytes = PyArray_NBYTES(p_fp_ImageArray);
    data = (uchar *) malloc(nbytes);
    memcpy(data, PyArray_DATA(p_fp_ImageArray), nbytes);
    copy = cv::Mat(PyArray_DIMS(p_fp_ImageArray)[0], PyArray_DIMS(p_fp_ImageArray)[1], image_in.type(), data);
    rend_fp_out = copy.clone();
    free(data);

    // (3) Handle frontal
    nbytes = PyArray_NBYTES(p_fr_ImageArray);
    data = (uchar *) malloc(nbytes);
    memcpy(data, PyArray_DATA(p_fr_ImageArray), nbytes);
    copy = cv::Mat(PyArray_DIMS(p_fr_ImageArray)[0], PyArray_DIMS(p_fr_ImageArray)[1], image_in.type(), data);
    rend_fr_out = copy.clone();
    free(data);

  } else {
    std::cout << "ERROR: Unknown number of poses returned by renderer" << std::endl;
    PyGILState_Release(gstate);
    return JANUS_UNKNOWN_ERROR;
  }

  Py_DECREF(pArgs);
  Py_DECREF(pTuple);

  std::cout << "Done with rendering" << std::endl;

  PyGILState_Release(gstate);
  return JANUS_SUCCESS;
}

janus_error
pythonGetPose(size_t imageHeight, size_t imageWidth, int nChannels, int cv_type, uint8_t *imageData,
              const std::vector<cv::Point2f> lm, int normalizationScale,
              cv::Mat &image_n, std::vector<cv::Point2f> &pose_landmarks ) {

  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  PyObject *image, *lmValue, *nsValue, *pArgs, *pValue, *pTuple;

  std::cout << "In pythonGetPose" << std::endl;
  if (imageHeight > 10000 || imageWidth > 10000) {
    std::cout << "Leaving pythonGetPose early--image too large" << std::endl;
    PyGILState_Release(gstate);
    return JANUS_UNKNOWN_ERROR;
  }

  npy_intp dim[3] = {(long int)imageHeight, (long int)imageWidth, nChannels};

  nsValue = PyInt_FromLong(long(normalizationScale));

  lmValue = PyTuple_New(2 * lm.size());
  //cout << "lm" << endl;                                                                                                                    
  int j=0;
  for (size_t i=0; i<lm.size(); ++i, j += 2) {
    //cout << lm[i].x << "," << lm[i].y << endl;                                                                                             
    pValue = PyFloat_FromDouble(double(lm[i].x));
    PyTuple_SetItem(lmValue, j, pValue);
    pValue = PyFloat_FromDouble(double(lm[i].y));
    PyTuple_SetItem(lmValue, j+1, pValue);
  }

  image = PyArray_SimpleNewFromData(3, dim, NPY_UINT8, imageData);

  pArgs = PyTuple_New(3);
  PyTuple_SetItem(pArgs, 0, image);
  PyTuple_SetItem(pArgs, 1, lmValue);
  PyTuple_SetItem(pArgs, 2, nsValue);

  pTuple = PyObject_CallObject(pGetPose, pArgs);
  if ((pTuple == NULL) || !PyTuple_Check(pTuple))  {
    PyErr_Print();
    std::cout << "call to getpos_no_init failed" << std::endl;
    Py_DECREF(pArgs);
    PyGILState_Release(gstate);
    return JANUS_UNKNOWN_ERROR;
  }

  PyArrayObject *pImageArray = (PyArrayObject *) PyTuple_GetItem(pTuple, 0);
  PyArrayObject *pLandmarks = (PyArrayObject *) PyTuple_GetItem(pTuple, 1);

  int nbytes = PyArray_NBYTES(pImageArray);
  uchar *data = (uchar *) malloc(nbytes);
  memcpy(data, PyArray_DATA(pImageArray), nbytes);
  cv::Mat copy = cv::Mat(PyArray_DIMS(pImageArray)[0], PyArray_DIMS(pImageArray)[1], cv_type, data);
  image_n = copy.clone();
  free(data);

  int num_landmarks = PyArray_DIMS(pLandmarks)[1];
  double *ldata = (double*) PyArray_DATA(pLandmarks);
  for (int i=0; i < num_landmarks; ++i) {
    pose_landmarks.push_back(cv::Point2f(ldata[i], ldata[i+num_landmarks]));
  }

  Py_DECREF(pArgs);
  Py_DECREF(pTuple);

  std::cout << "Done pythonGetPose" << std::endl;
  PyGILState_Release(gstate);
  return JANUS_SUCCESS;
}

janus_error
pythonFinalize() {
  PyEval_RestoreThread(_threadstate_save);

  Py_XDECREF(poseModule);
  Py_Finalize();
  return JANUS_SUCCESS;
}

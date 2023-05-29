#pragma once
#include "vvas_xdpupriv.hpp"

#include <vitis/ai/yolovx.hpp>
#include <vitis/ai/nnpp/yolovx.hpp>

using namespace std;
using namespace cv;

class vvas_xyolovx:public vvas_xdpumodel
{

  int log_level = 0;
  std::unique_ptr < vitis::ai::YOLOvX > model;

public:

  vvas_xyolovx (vvas_xkpriv * kpriv, const std::string & model_name,
      bool need_preprocess);

  virtual int run (vvas_xkpriv * kpriv, std::vector<cv::Mat>& images,
      GstInferencePrediction **predictions);

  virtual int requiredwidth (void);
  virtual int requiredheight (void);
  virtual int supportedbatchsz (void);
  virtual int close (void);

  virtual ~vvas_xyolovx ();
};

#ifndef ANN_CLASSIFIER_H_
#define ANN_CLASSIFIER_H_

#include "CircularBuffer.h"
#include "Data.h"

class ANNClassifier {

public:
  ANNClassifier() : _num(0) {}
  float *Predict(Data data);

private:
  CircularBuffer<Data, 2> _input;
  int _num;

  struct XStep {
    float xoffset[8];
    float gain[8];
    float ymin;
  };

  struct YStep {
    float xoffset[3];
    float gain[3];
    float ymin;
  };

  static const XStep X1_STEP1;
  static const YStep Y1_STEP1;
  static const float B1[10];
  static const float B2[3];
  static const float IW1_1[10][16];
  static const float LW2_1[3][10];
};

#endif

/*
The MIT License (MIT)

Copyright (c) <2016> <David Hasenfratz>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

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

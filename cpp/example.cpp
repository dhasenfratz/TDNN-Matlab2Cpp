/*
The MIT License (MIT)

Copyright (c) 2017 David Hasenfratz

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

#include <iostream>
#include "Data.h"
#include "ANNClassifier.h"

using namespace std;

static const float POLLUTION_DATASET[5][8] = {
  {72.38, 29.20, 11.51, 3.37, 9.64, 45.79, 6.69, 72.72},
  {67.19, 67.51, 8.92, 2.59, 10.05, 43.90, 6.83, 49.60},
  {62.94, 61.42, 9.48, 3.29, 7.80, 32.18, 4.98, 55.68},
  {72.49, 58.99, 10.28, 3.04, 13.39, 40.43, 9.25, 55.16},
  {74.25, 34.80, 10.57, 3.39, 11.90, 48.53, 9.15, 66.02}
};

int main(int argc, char* argv[]) {

  // Test the classifier with the first five inputs of Matlab's dataset
  // pollution_dataset, which was used to train the neural network.
  Data data;
  ANNClassifier classifier;
  float *prediction;

  for (int d = 0; d < 5; d++) {
    for (int i = 0; i < 8; i++) {
      data.value[i] = POLLUTION_DATASET[d][i];
    }
    prediction = classifier.Predict(data);
    cout << "Prediction " << prediction[0] << ", " << prediction[1] << ", " << prediction[2] << endl;
  }
  return 0;
}

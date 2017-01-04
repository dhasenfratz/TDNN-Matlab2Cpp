% The MIT License (MIT)
%
% Copyright (c) 2017 David Hasenfratz
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.

function createSourceFile(net)

  fid=fopen('../cpp/ANNClassifier.cpp', 'w');
  fprintf(fid,'#include <cmath>\n');
  fprintf(fid,'#include "ANNClassifier.h"\n\n');

  fprintf(fid,'const ANNClassifier::XStep ANNClassifier::X1_STEP1 = {\n');
  fprintf(fid, '  {');
  fprintf(fid, '%f, ', net.inputs{1}.processSettings{1}.xoffset(1:end-1));
  fprintf(fid, '%f},\n', net.inputs{1}.processSettings{1}.xoffset(end));
  fprintf(fid, '  {');
  fprintf(fid, '%f, ', net.inputs{1}.processSettings{1}.gain(1:end-1));
  fprintf(fid, '%f},\n', net.inputs{1}.processSettings{1}.gain(end));
  fprintf(fid, '  %f\n};\n\n', net.inputs{1}.processSettings{1}.ymin);

  fprintf(fid,'const ANNClassifier::YStep ANNClassifier::Y1_STEP1 = {\n');
  fprintf(fid, '  {');
  fprintf(fid, '%f, ', net.outputs{2}.processSettings{1}.xoffset(1:end-1));
  fprintf(fid, '%f},\n', net.outputs{2}.processSettings{1}.xoffset(end));
  fprintf(fid, '  {');
  fprintf(fid, '%f, ', net.outputs{2}.processSettings{1}.gain(1:end-1));
  fprintf(fid, '%f},\n', net.outputs{2}.processSettings{1}.gain(end));
  fprintf(fid, '  %f\n};\n\n', net.outputs{2}.processSettings{1}.ymin);

  fprintf(fid,'const float ANNClassifier::B1[%d] = {', numel(net.b{1}));
  fprintf(fid, '%f, ', net.b{1}(1:end-1));
  fprintf(fid, '%f};\n\n', net.b{1}(end));

  fprintf(fid,'const float ANNClassifier::B2[%d] = {', numel(net.b{2}));
  fprintf(fid, '%f, ', net.b{2}(1:end-1));
  fprintf(fid, '%f};\n\n', net.b{2}(end));

  fprintf(fid,'const float ANNClassifier::IW1_1[%d][%d] = {\n', size(net.IW{1}));
  s = size(net.IW{1});
  for i=1:s(1)
    fprintf(fid, '  {');
    fprintf(fid, '%f, ', net.IW{1}(i,1:end-1));
    if i == s(1)
      fprintf(fid, '%f}};\n\n', net.IW{1}(i,end));
    else
      fprintf(fid, '%f},\n', net.IW{1}(i,end));
    end
  end

  fprintf(fid,'const float ANNClassifier::LW2_1[%d][%d] = {\n', size(net.LW{2}));
  s = size(net.LW{2});
  for i=1:s(1)
    fprintf(fid, '  {');
    fprintf(fid, '%f, ', net.LW{2}(i,1:end-1));
    if i == s(1)
      fprintf(fid, '%f}};\n\n', net.LW{2}(i,end));
    else
      fprintf(fid, '%f},\n', net.LW{2}(i,end));
    end
  end

  fprintf(fid,'float *ANNClassifier::Predict(Data data) {\n');
  fprintf(fid,'  static float prediction[%d];\n', net.outputs{2}.size);
  fprintf(fid,'  float sum = 0.0f;\n\n');
  fprintf(fid,'  for (int i = 0; i < %d; i++) {\n', net.outputs{2}.size);
  fprintf(fid,'    prediction[i] = 0.0f;\n');
  fprintf(fid,'  }\n\n');
  fprintf(fid,'  for (int i = 0; i < %d; i++) {\n', net.inputs{1}.size);
  fprintf(fid,'    data.value[i] = (data.value[i] - X1_STEP1.xoffset[i]) * X1_STEP1.gain[i] + X1_STEP1.ymin;\n');
  fprintf(fid,'  }\n\n');
  fprintf(fid,'  _num++;\n');
  fprintf(fid,'  if (_num > %d) {\n\n', net.numInputDelays);
  fprintf(fid,'    for (int n = 0; n < %d; n++) {\n', numel(net.b{1}));
  fprintf(fid,'      sum = 0.0f;\n');
  fprintf(fid,'      for (int i = %d-1; i >= 0; i--) {\n', net.numInputDelays);
  fprintf(fid,'        Data temp = _input.get(i);\n');
  fprintf(fid,'        for (int j = 0; j < %d; j++) {\n', net.inputs{1}.size);
  fprintf(fid,'          sum += temp.value[j] * IW1_1[n][((%d-1)-i)*%d+j];\n', net.numInputDelays, net.inputs{1}.size);
  fprintf(fid,'        }\n');
  fprintf(fid,'      }\n');
  fprintf(fid,'      sum += B1[n];\n');
  fprintf(fid,'      sum = 2 / (1 + expf(-2*sum)) - 1;\n\n');
  fprintf(fid,'      for (int i = 0; i < %d; i++) {\n', numel(net.b{2}));
  fprintf(fid,'        prediction[i] += sum * LW2_1[i][n];\n');
  fprintf(fid,'      }\n');
  fprintf(fid,'    }\n');
  fprintf(fid,'    for (int i = 0; i < %d; i++) {\n', numel(net.b{2}));
  fprintf(fid,'      prediction[i] += B2[i];\n');
  fprintf(fid,'      prediction[i] = (prediction[i] - Y1_STEP1.ymin) / Y1_STEP1.gain[i] + Y1_STEP1.xoffset[i];\n');
  fprintf(fid,'    }\n');
  fprintf(fid,'  }\n\n');
  fprintf(fid,'  _input.add(data);\n\n');
  fprintf(fid,'  return prediction;\n');
  fprintf(fid,'}\n');

  fclose(fid);
end

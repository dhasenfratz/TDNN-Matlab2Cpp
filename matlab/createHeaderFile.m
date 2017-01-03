function createHeaderFile(net)

  fid=fopen('../cpp/ANNClassifier.h', 'w');
  fprintf(fid,'#ifndef ANN_CLASSIFIER_H_\n');
  fprintf(fid,'#define ANN_CLASSIFIER_H_\n\n');
  fprintf(fid,'#include "CircularBuffer.h"\n');
  fprintf(fid,'#include "Data.h"\n\n');
  fprintf(fid,'class ANNClassifier {\n\n');
  fprintf(fid,'public:\n');
  fprintf(fid,'  ANNClassifier() : _num(0) {}\n');
  fprintf(fid,'  float *Predict(Data data);\n\n');
  fprintf(fid,'private:\n');
  fprintf(fid,'  CircularBuffer<Data, %d> _input;\n', net.numInputDelays);
  fprintf(fid,'  int _num;\n\n');
  fprintf(fid,'  struct XStep {\n');
  fprintf(fid,'    float xoffset[%d];\n', net.inputs{1}.size);
  fprintf(fid,'    float gain[%d];\n', net.inputs{1}.size);
  fprintf(fid,'    float ymin;\n');
  fprintf(fid,'  };\n\n');
  fprintf(fid,'  struct YStep {\n');
  fprintf(fid,'    float xoffset[%d];\n', net.outputs{2}.size);
  fprintf(fid,'    float gain[%d];\n', net.outputs{2}.size);
  fprintf(fid,'    float ymin;\n');
  fprintf(fid,'  };\n\n');
  fprintf(fid,'  static const XStep X1_STEP1;\n');
  fprintf(fid,'  static const YStep Y1_STEP1;\n');
  fprintf(fid,'  static const float B1[%d];\n', numel(net.b{1}));
  fprintf(fid,'  static const float B2[%d];\n', numel(net.b{2}));
  fprintf(fid,'  static const float IW1_1[%d][%d];\n', size(net.IW{1}));
  fprintf(fid,'  static const float LW2_1[%d][%d];\n', size(net.LW{2}));
  fprintf(fid,'};\n\n');
  fprintf(fid,'#endif\n');
  fclose(fid);

  fid=fopen('../cpp/Data.h', 'w');
  fprintf(fid,'#ifndef DATA_H_\n');
  fprintf(fid,'#define DATA_H_\n\n');
  fprintf(fid,'struct Data {\n');
  fprintf(fid,'  float value[%d];\n', net.inputs{1}.size);
  fprintf(fid,'};\n\n');
  fprintf(fid,'#endif\n');
  fclose(fid);
end
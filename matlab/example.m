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

% Make results repeatable.
rng(1);

% Pollution data set:
%  - PollutionInputs: every input consists of 8 variables.
%  - PollutionTargets: every target consists of 3 variables.
load pollution_dataset

% Training function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a time delay network.
inputDelays = 1:2;
hiddenLayerSize = 10;
net = timedelaynet(inputDelays, hiddenLayerSize, trainFcn);

% Prepare data for training.
[x,xi,ai,t] = preparets(net, pollutionInputs, pollutionTargets);

% Setup Division of Data for training, validation, testing.
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the network.
[net,tr] = train(net,x,t,xi,ai);

% Create C++ implementation of the trained network.
extractNeuralNetwork(net);

% The results of the predictions for input 3, 4, and 5 can be compared
% to the prediction of the neural network implemented in C++, see
% example.cpp.
% Here, we compare it to the targets used to train the neural network.
prediction = net(pollutionInputs(3:5), pollutionInputs(1:2), ai);
for i=1:3
  fprintf('Prediction %.2f, %.2f, %.2f and target %.2f, %.2f, %.2f\n', ...
          cell2mat(prediction(i)), cell2mat(pollutionTargets(i+2)));
end

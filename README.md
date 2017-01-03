neuralNetworkMat2Cpp
====================

This program takes a time-delay neural network trained in Matlab using timedelaynet of Matlab's Neural Network Toolbox and creates the header and source file of a new C++ class ANNClassifier implementing the trained time-delay neural network.

This is useful for deploying code developed in Matlab into embedded
applications where Matlab is not available.

Usage
-----

The Matlab function extractNeuralNetwork takes as input a trained time-delay neural network and outputs the header files Data.h and ANNClassifier.h and the source file ANNClassifier.cpp. For a detailed usage example see the files example.m and example.cpp.

License
------------

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

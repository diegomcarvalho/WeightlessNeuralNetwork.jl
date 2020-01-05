# WeightlessNeuralNetwork.jl Package

## Introduction

Implements the WiSARD Neural Network (Wilkes, Stonham and Aleksander Recognition Device). It is composed of a set of discriminators, often called as classifiers[1]. This specific implementation is based on the description provided by [2], and the zero_skip and padded_retina were based on the IAZero implementation.

2019,2020 (@) Diego Carvalho - d.carvalho@ieee.org

> [1]: I. Aleksander, W. Thomas, P. Bowden, WISARD a radical step forward in image recognition, Sens. Rev. 4 (1984) 120–124.

> [2]: M. De Gregorio and M. Giordano, “Cloning DRASiW systems via memory transfer,” Neurocomputing, vol. 192, pp. 115–127, 2016.

## TODO

- [X] New feature: parallel discriminators --- Implement parallel discriminators.
- [ ] Refactor wisard to access data by indices --- The parallel version of Drasiw is super fast due to the indices acessess of data. So, the Wisard class should be rewriten to incorporate the advantages. 
- [ ] New feature: a test module --- Implement the test module. It should scan addresses, size, classes, etc.
- [ ] New feature: random generation control --- Provide a way to control the random generation. Each wisard net should have a random generation stream and this information must propagate through layers since Discriminator must use on the retina mapping.
- [ ] New feature: record wisard state --- Should implement IO functions in order to record the current wisard training state. Perhaps, it should use JSON.
- [x] New feature: confidence level --- Should calculate the confidence level for each wisard output.
- [x] Bug fix (1) --- The code below can crash when b reaches m during bleaching, and winners is greater than one (line 83).
- [ ] Implement ClusWisard
- [ ] Implement MageNet

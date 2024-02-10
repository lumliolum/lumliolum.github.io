---
layout: post
title: TWO STAGE OBJECT DETECTORS
---

In this post, we will discuss about 2 stage object detectors Fast-RCNN, Faster-RCNN and their ideas. I will also give a breif introduction to RCNN but will not go into much detail.

2 stage detectors contains 2 stages for classifying and detecting objects. Typically, the first network will give the region proposals (the region where object might exist) and second network will take the region proposals as input and detects the object.

### RCNN

RCNN contains 3 modules
- The first one generates the region proposal. This stage doesn't use any neural network, but uses an algorithm called as selective search. Typically generates 2000 proposals for each image
- CNN which is used as feature extractor. The input to this is a region proposal (from above) and output is a vector (representing the region proposal).
- The third module is a linear SVM and bounding box regressor that wil take the feature vector from the above step and predict for each class.

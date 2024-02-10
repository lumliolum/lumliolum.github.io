---
layout: post
title: TWO STAGE OBJECT DETECTORS
---

In this post, we will discuss about 2 stage object detectors Fast-RCNN, Faster-RCNN and their ideas. I will also give a breif introduction to RCNN but will not go into much detail.

2 stage detectors contains 2 stages for classifying and detecting objects. Typically, the first network will give the region proposals (the region where object might exist) and second network will take the region proposals as input and detects the object.

## RCNN

RCNN contains 3 modules

- The first one generates the region proposal. This stage doesn't use any neural network, but uses an algorithm called as selective search. Typically generates 2000 proposals for each image
- CNN which is used as feature extractor. The input to this is a region proposal (from above) and output is a vector (representing the region proposal).
- The third module is a linear SVM and bounding box regressor that wil take the feature vector from the above step and predict for each class.

Note : Whenever region proposal is mentioned, assume that it is rectangular. This rectangular is cropped from original image, then we do the processing on the smaller part.

### FEATURE EXTRACTION

I will not discuss much about selective search (I don't understand how that thing works) but assumming that we have region proposals, we resize them to (227, 227) and then passed to CNN network (Alexnet, vggnet etc) to get the vector representation of the proposal.

The feature extractor is usually pretrained on image-net and then finetuned on region proposals. One question should naturally come is how do we create training data for region proposal ?, how do we know what is ground-truth for any region proposal.

please check $f(x) = x^2$

$$
f(x) = x^2\\
\nabla_{x} f= 2x
$$

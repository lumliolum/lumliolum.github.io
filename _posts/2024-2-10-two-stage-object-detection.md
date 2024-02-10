---
layout: post
title: TWO STAGE OBJECT DETECTION
---

In this post, we will discuss about 2 stage object detectors Fast-RCNN, Faster-RCNN and their ideas. I will also give a breif introduction to RCNN but will not go into much detail.

2 stage detectors contains 2 stages for classifying and detecting objects. Typically, the first network will give the region proposals (the region where object might exist) and second network will take the region proposals as input and detects the object.

## RCNN

[RCNN](https://arxiv.org/abs/1311.2524) contains 3 modules

- The first one generates the region proposal. This stage doesn't use any neural network, but uses an algorithm called as selective search. Typically generates 2000 proposals for each image
- CNN which is used as feature extractor. The input to this is a region proposal (from above) and output is a vector (representing the region proposal).
- The third module is a linear SVM and bounding box regressor that wil take the feature vector from the above step and predict for each class.

Note : Whenever region proposal is mentioned, assume that it is rectangular. This rectangular is cropped from original image, then we do the processing on the smaller part.

### FEATURE EXTRACTION

I will not discuss much about selective search (I don't understand how that thing works) but assumming that we have region proposals, we resize them to (227, 227) and then passed to CNN network (Alexnet, vggnet etc) to get the vector representation of the proposal.

The feature extractor is usually pretrained on image-net and then finetuned on region proposals. The finetuning stage is called domain specific fine-tuning.

One question should naturally come is how do we create training data for region proposals ?, how do we know what is ground-truth for any region proposal.

So while finetuning, we remove the last layer of feature extractor (if its pretrained on image-net, we usually have it as 1000 neurons) and replace it with $(N+1)$ neurons where $N$ denotes the number of classes (+1 is for the background class). This answers our 2nd question.

The 1st question will occur in Fast-RCNN, Faster-RCNN as well. I will discuss in detail over there.

### OBJECT CATEGORY CLASSIFIER or LINEAR SVM

Once the feature extractor is trained, we train SVM's for each class. The input to the SVM is the feature vector from above step (usually of size 4096)

Same question should occur here as well, how do we create training dataset for SVM ?
I will skip the answer to this question.

Another qusetion should occur is why are we using SVM ?. Anyway the feature extractor was trained using $(N+1)$ neurons as last layer, so it can tell for each proposal the class it belongs to.

Authors tried this and found that performance on VOC dropped from 54% to 50% on mAP. So they kept the idea of SVM (See appendix B of the paper for more details).

### TESTING

The above section concludes on training. The following steps describe how the testing is done for an image in RCNN

- Run selective search on test image to get 2000 proposals. Resize all the proposals to shape (227, 227)
- Run the forward propogation for each proposal through finetuned CNN to get the extraced features (size 4096)
- Pass each feature to N SVM's to get the scores and select the class with highest score. This way we will get predicted classes for all the proposals
- Per class bounding box regressor is used to predict the box co-ordinates (skipped this part)
- As the last step, we apply class wise nms (non maximum supression) to remove the overlapping proposals.

With this we close RCNN and move to Fast-RCNN

## FAST RCNN

[Fast RCNN](https://arxiv.org/abs/1504.08083?so) was introduced to make RCNN fast (as name suggests) and also it performs better that RCNN

Before going I want to address some points

- I know that RCNN was not covered in detail. Anyone reading it for the first time may not understand the section of RCNN. The reason I wrote is just to familiarize yourself with the terminology.
- The reason I skipped important parts of RCNN is because I feel that they are not useful. Also from my pov, RCNN is not a simple and direct model and has lot of complications. In simple words, it's chaos.
- In terms of that fast rcnn is more simple and direct. Lot of back-forth steps from RCNN is removed.

Fast RCNN is created from RCNN, but it's fast. RCNN takes 9.8 seconds while fast RCNN takes 0.1 seconds at test time (See table 4)

One of the main reason why RCNN is slow is because feature extraction is done for each proposal. For example, if we have 100 proposals, then each proposal is cropped and passed to CNN for feature extraction. This is where fast-rcnn optimizes

The changes introduced by fast-rcnn are

- Introduces the ROI layer which will enable it to calcuate the feature vectors for all proposals at once.
- Removes SVM and both classification and regression are done using dense layer
- Proposes a 1 step training pipeline where both classification and regression loss is optimized compare to RCNN where classification and regression were trained seperately.

### ARCHITECTURE

The architecure for fast-rcnn is given below

![fast-rcnn-arch.png](/images/2-stage-object-detection/fast-rcnn-arch.png)

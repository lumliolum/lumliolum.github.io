---
layout: post
title: Faster RCNN
date: 2025-05-14
---

Here we will discuss about faster rcnn in more detail.

## INTRODUCTION

Faster RCNN was introduced in less than a year after Fast RCNN. It is better, both in terms of speed and performance than Fast RCNN.

Even though the original paper was released 5-6 years back, there were lot of updates on top of original work. One of them was Mask RCNN which was built on top of faster RCNN. Mask RCNN is basically used for instance segmentation but it also introduces ROI Align layer in the place of ROI Pool layer.

Going through each of the update and also mentioning the historical timeline will be difficult (another defence I have is that I don't know all the updates). Instead what I will do here is mention the details of what I feel the stable version of faster rcnn.

The major changes with respect to fast-rcnn are

- Introduction of Region Proposal Network (RPN) to generate proposals.
- ROI Align layer to replace ROI Pool layer. As mentioned this was not introduced in the original paper but introduced in Mask RCNN paper.

If we measure the inference time of Fast RCNN, it usually takes 2 seconds for selective search (proposal generation) and 0.1 seconds for detection part. So it nearly takes 2.1 seconds per image and majority is from selective search. The detection part uses CNN which will take the advantage of GPU's while selective search uses CPU's (one other reason why selective search is slow).

To generate proposals quickly, the Region Proposal Network (RPN) was introduced, which utilizes a CNN to produce region proposals. An additional advantage of using a CNN for this task is that it enables end-to-end training of the entire model, including both the RPN and the detection network.

Faster RCNN contains a full convolutional backbone which take an image and generate feature map. This feature map is then passed to RPN network which gives bounding boxes (proposals) and objectness score for each box. The same feature map and the predicted proposals are passed to Fast RCNN detection network which is basically RoIAlign (as of now think this as RoIPool) + FC layers to predict bounding box and their class.

The architecture of Faster RCNN is given below

![faster-rcnn-architecture.png](/images/2-stage-object-detection/faster-rcnn-architecture.png)

### RPN

The purpose of RPN is to take an image and output set of rectangular object proposals with objectness score.

---
layout: post
title: Faster RCNN
excerpt: Details about Faster RCNN.
---

Here we will discuss about faster rcnn in more detail.

## FASTER RCNN

Faster RCNN was introduced in less than a year after Fast RCNN. It is better, both in terms of speed and performance than Fast RCNN.

Even though the original paper was released, there were lot of updates on top of original work. One of them was Mask RCNN which was built on top of faster RCNN. Mask RCNN is basically used for instance segmentation but it also introduces ROI Align layer in the place of ROI Pool layer.

Going through each of the update and also mentioning the historical timeline will be difficult (another defence I have is that I don't know all the updates). Instead what I will do here is mention the details of what I feel the stable version of faster rcnn.

The major changes with respect to fast-rcnn are

- Introduction of Region Proposal Network (RPN) to generate proposals.
- ROI Align layer to replace ROI Pool layer. As mentioned this was not introduced in the original paper but we will discuss this here.

There are some other minor changes which will come up.

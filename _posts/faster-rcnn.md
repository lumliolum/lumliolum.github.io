---
layout: post
title: Faster RCNN
date: 2025-05-14
---

Here we will discuss about Faster RCNN in more detail.

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

![faster-rcnn-architecture](/images/2-stage-object-detection/faster-rcnn-architecture.png)

Observe that the convoluion backbone is same for RPN and detection network.

### RPN

The purpose of RPN is to take an image and output set of rectangular object proposals with objectness score. Following the same notation like last time, let's say the input image is of shape $(3, H, W)$ and output from convolution backbone be $(C, H_{f}, W_{f})$ (feature map).

To generate proposals, the approach involves predicting $k$ boxes at each location of the feature map. So there will be $H_{f}W_{f}k$ boxes, where each box includes 4 regression values for coordinates and 2 classification scores indicating the presence or absence of an object. (The authors use a 2-class softmax for classification, but this can alternatively be implemented as a single-class prediction with a sigmoid activation).

So at each location we will have $4k$ regression scores and $2k$ classification scores. The way this is done is by passing the feature map output by a $3 \times 3$ convolution layer (stride $1$ and padding $1$.) and then two $1 \times 1$ convolution layers - one for generating classification scores and the other for regression outputs. The figure below illustrates the architecture of the RPN.

![rpn-architecture](/images/2-stage-object-detection/rpn-architecture.png)

Till now we understood how to predict or generate proposals using the output from convolution backbone. The pressing question is how do we train RPN ?

#### TRAINING RPN

Here, I will outline the procedure as described by the authors in the paper. It's worth reflecting on why this approach was chosen and whether there might be more effective alternatives.

Given an image and a set of ground truth bounding boxes, and assuming we predict one box at each location (using the procedure described above), how do we compute the loss? More specifically, how do we determine which predicted box should be matched with which ground truth box? Take a moment to consider this carefully — I think it’s not a straightforward problem and coming up with a solution is also not easy.

The inconvient part in the above question is how do we determine which predicted box should be matched with which ground truth box. To do this, at each location we will have $k$ reference boxes which are called as anchors. These anchors are fixed and independent of predictions. We'll first briefly explain how the coordinates of anchor boxes are determined, and then relate this to how RPN training works.

Each location on the feature map serves as the center for a set of anchors. As the image size is $(H, W)$ and feature map size is $(H_{f}, W_{f})$, the center of anchors are located at $\left(\left(m_{1} + \frac{1}{2}\right)\frac{H}{H_{f}}, \left(m_{2} + \frac{1}{2}\right)\frac{W}{W_{f}}\right)$ where $0 \leq m_{1} \leq H_{f} - 1, 0 \leq m_{2} \leq W_{f} - 1$ are integers. Below image shows the anchor centers for a sample image.

![anchor-centers](/images/2-stage-object-detection/anchor-centers.png)

At each center, we will have $k$ anchor boxes of different box areas (scales) and aspect ratios (This is also the reason why we predict $k$ boxes). For examples if we choose $3$ box areas as $128^{2}, 256^{2}, 512^{2}$ and $3$ aspect ratios $1:1, 1:2, 2:1$ implying we end up with $k = 3 \times 3 = 9$ anchor boxes at each location. Aspect ratio is defined as ratio of height to width. Let's do a small example to get more understanding of the same.

Let's suppose the input image size is $(600, 795)$ and feature map outpuut size is $(38, 50)$. So the spacing in each direction will be

\begin{align}
    S_{h} = \frac{600}{38} = 15.79, S_{w} = \frac{795}{50} = 15.90
\end{align}

So one of the anchor center will be at location $(7.89, 7.95)$ and if we choose the scale as $128^{2}$ and aspect ratio as $1:1, 1:2, 2:1$ then the height and width of anchors will be $(128, 128), (90.51, 181.02), (181.02, 90.51)$.

During training, each anchor box is matched to a ground truth box and categorized as positive, negative, or ignored. A positive label is assigned to two kind of anchors

1. The anchor/anchors with highest IoU overlap with a ground truth box
2. An anchor that has an IoU overlap higher than $0.7$ with any ground truth box.

Also a negative label is assigned to an anchor if it's IoU is lower than $0.3$ for all ground truth boxes. Anchors that are neither positive nor negative are simply ignored(These are rules taken from the paper).

So at each location, we have $k$ predicted boxes and $k$ anchors boxes and each anchor box has a label which is positive, negative or ignored. Defining the classification loss using this information will be easy. Before we delve into the computation of the regression loss, let’s first introduce some notations. Let the prediction box be $$(x, y, h, w)$$ and anchor box be $$(x_{a}, y_{a}, h_{a}, w_{a})$$ and ground truth box be {::nomarkdown}$(x^{*}, y^{*}, h^{*}, w^{*})${:/}, then

{::nomarkdown}
\begin{align*}
    t_{x} = \frac{x - x_{a}}{w_{a}}, & t_{y} = \frac{y - y_{a}}{h_{a}} \\
    t_{w} = \log\left(\frac{w}{w_{a}}\right), & t_{h} = \log\left(\frac{h}{h_{a}}\right) \\
    t^{*}_{x} = \frac{x^{*} - x_{a}}{w_{a}}, & t^{*}_{y} = \frac{y^{*} - y_{a}}{h_{a}} \\
    t^{*}_{w} = \log\left(\frac{w^{*}}{w_{a}}\right), & t^{*}_{h} = \log\left(\frac{h^{*}}{h_{a}}\right)
\end{align*}
{:/}

Using the same loss function as in Fast RCNN, the regression loss will be

{::nomarkdown}
\begin{align}
    L_{R}(t, t^{*}) = \sum_{i \in \{x,y,w,h\}} f_{s}(t_{i} - t_{i}^{*})
\end{align}
{:/}

where $f_{s}$ is defined as

{::nomarkdown}
\begin{align*}
f_{s}(x) =
    \begin{cases}
        0.5x^2 & \text{if $\lvert x \rvert < 0.5$ } \\
        \lvert x \rvert - 0.5 & \text{otherwise}
    \end{cases}
\end{align*}
{:/}

The RPN loss for one image will

{::nomarkdown}
\begin{align*}
    L_{j} = \frac{1}{N_{C}}\sum_{i}L_{C}(p_{i}, p_{i}^{*}) + \frac{\lambda}{N_{R}}\sum_{i}p_{i}^{*}L_{reg}(t_{i}, t_{i}^{*})
\end{align*}
{:/}

where $p_{i}$ is the predicted probability of anchor $i$ being positive, {::nomarkdown}$p_{i}^{*}${:/} is 1 is anchor is positive and 0 if anchor is negative. Observe that regression loss is only computed for anchors which are positive. Each mini batch arises from a single image that contains positive and negative example anchors. Authors have used batch size of 256 where positive and negative anchors are in ratio of $1:1$. During training, anchor boxes that extend beyond the image boundaries are ignored and do not contribute to the RPN loss. However, during testing, the model may predict boxes that cross the image boundaries - these are simply clipped to fit within the image.

Some RPN proposals highly overlap with each other, and to reduce the redudancy non maximum supression (NMS) is applied based on their classification scores. After NMS, top-$N$ ranked proposals are used for detection.

### ROI ALIGN

RoI Align was first introduced in the Mask R-CNN paper as a replacement for RoI Pooling to improve mask prediction and solve the alignment issue. Although it was originally motivated by instance segmentation, RoI Align is now commonly used in object detection as well, replacing RoI Pool for better performance.

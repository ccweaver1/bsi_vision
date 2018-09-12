# BSI Vision

This work uses deep convolutional neural networks and other ML techniques to extract player detection and mapping data from NHL broadcasts.

## Objective

To make an open source platform that can injest broadcast footage of hockey games and output player mappings as well as other useful information harvested from these frames.

This is accomplished in 3 main stages: data collection, object detection and homography generation.

## Data Collection

Generating labelled data for the training of the relevant networks meant finding/designing labelling tools as well as an efficient data pipeline for storing and updating these pieces of data in a distributed manner.

The object detection network requires bounding boxes and labels for relevant hockey objects.  For our purposes, these objects were: hockey_player, hockey_ref, hockey_goalie, hockey_blue_line, hockey_middle_line, hockey_end_line, hockey_goal and hockey_scorebug.  For generating these labels and boxes, [labelImg] was used. (https://github.com/tzutalin/labelImg)  This made it simpler to drag the boxes, but the annotation files it generated were in XML (so I converted them to JSON). 

In order to address the challenge of predicting homography matrices when given a frame, I collected frames portraying a range of camera angles and hockey scenes.  These were labelled with a 9-wide homography vector which was generated using a custom built python tool which allowed to drag lines between corresponding points (see below).

![alt text](https://raw.githubusercontent.com/ccweaver1/bsi_vision/master/demo/matching.png)


The data pipeline was built using 
## Object Detection

Object detection is accomplished using a [retinanet] architechture with a ResNet backbone.  This network is 
(https://github.com/fizyr/keras-retinanet)

### In action




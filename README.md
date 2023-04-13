# PPNet

![image](Images/PPNet.png)

## A Pyramidal Predictive Network for Video Prediction Base on Predictive Coding Theory 

Visual-frame prediction is a pixel-dense prediction task that infers future frames from past frames. Lacking of appearance details, low prediction accuracy and high computational overhead are still major problems with current models or methods. In this paper, we propose a novel neural network model inspired by the well-known predictive coding theory to deal with the problems. Predictive coding provides an interesting and reliable computational framework, which will be combined with other theories such as the cerebral cortex at different level oscillates at different frequencies, to design an efficient and reliable predictive network model for visual-frame prediction. Specifically, the model is composed of a series of recurrent and convolutional units forming the top-down and bottom-up streams, respectively. The update frequency of neural units on each of the layer decreases with the increasing of network levels, which results in neurons of higher-level can capture information in longer time dimensions. According to the experimental results, this model shows better compactness and comparable predictive performance with existing works, implying lower computational cost and higher prediction accuracy.




# Multi-modal deep learning algorithm to detect mastoiditis on mastoid series
 - Convolutional neural networks for diagnosing mastoiditis
   1. data/setup.py   (Upload Mastoid AP & Lateral view, label)
   2. models/network.py   (CNN for AP view, Lateral view, Multiple views)
   3. runs/main1.py   (Training for single view)
   4. runs/main2.py   (Training for multiple views)
   5. tf_utils/tboard.py   (Related to Tensorboard output)
   
## 1. Convolution neural network for single view
   <p align="center">
      <img src="https://user-images.githubusercontent.com/49828672/102782145-31a9a680-43dc-11eb-9ca8-250bcde9bc9c.png" width=50% height=50% img align="center"> 
   </p>

## 2. Convolution neural network for multiple views
   <p align="center">
      <img src="https://user-images.githubusercontent.com/49828672/102782148-32dad380-43dc-11eb-88e6-5765c9558625.png" width=50% height=50% img align="center"> 
   </p>

## 3. Class activation mapping for detecting mastoiditis
   <p align="center">
      <img src="https://user-images.githubusercontent.com/49828672/102781780-97496300-43db-11eb-816b-4304d0beec81.png" width=50% height=50% img align="center"> 
   </p>



##### Reference: Lee, K., Ryoo, I., Choi, D., Sunwoo, L., You, S., & Jung, H.I. (2020). Performance of deep learning to detect mastoiditis using multiple conventional radiographs of mastoid. PLoS ONE, 15.

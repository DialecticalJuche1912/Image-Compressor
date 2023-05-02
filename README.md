# Image-Compressor
## Functionalities: 
This app performs image compression using Singular Value Decomposition (SVD) in Python. It can handle both grayscale and colored images, which uses the SVD method to reconstruct images into 64 matrices which reduces the image size to a fraction of original size. 

## Tools Required
To run this image compressor, you will need to have 

Python 3.7 or later
A text editor or an IDE (like Visual Studio Code)
Python libaries needed: numpy, PIL, matplotlib, and operating interfaces such as os, io and requests 

## Installation and Useage 

1. Install Python 3.7 or later, if not already installed.

2. Install the required Python libraries:
`pip install numpy Pillow requests` and 
`pip install matplotlib`

3. Open `imagecompression.py` for grayscale version or `imompressRGB.py` for full color version in your preferred IDE 
4. Customize the url variable with the URL of the image you'd like to compress
5. Click `run` feature which will execute the script and display the compressed images with various ranks along with the singular values and their cumulative sum plots.


## Acknowledgement 
This app is inspired by the SVD algorithm from John Krohn's Udemy course "Mathematical Foundations of Machine Learning" 
The Matrix calculation technique used is learned from Math 136 @ University of Waterloo 

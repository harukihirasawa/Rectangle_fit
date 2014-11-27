# Rectangle_fit.py README
#### Haruki Hirasawa 
##### November 27, 2014
#####Memorial University of Newfoundland, Dept. of Physics and Physical Oceanography
### Introduction
**rectangle_fit.py** is a python code designed for use in analyzing images of rod-like colloids. I was written by Haruki Hirasawa under the supervision of Dr. Anand Yethiraj of the Memorial University of Newfoundland, Department of Physics and Physical Oceanography.

### What does Rectangle_fit do?
**rectangle_fit.py** analyzes images of rod colloids as follows:
1. Find an appropriate threshold value for the image
2. Take the points whose intensity surpasses the threshold
3. Find the regions of connected bright pixels
4. Ignoring small regions, find the centroid and "major axis" of each pixel blob
5. Fit a bounding rectangle to the pixel blob using the technique described by D. Chaudhuri et. al. in "_Finding best-fitted rectangle for regions using a bisection method_" (2014)
6. Classify each rectangle based on a variety of criteria into three categories: One rod, Two Rods, Three or more Rods
7. By using the profiles of the rectangles along the major and minor axes, split any long rectangle as well as optimized any one rod rectangles.
8. Overlay fitted rectangles onto the original image and print
9. Calculate and print metadata to a .dat file
10. Output data into a .csv file
11. Print a number of histograms to characterize the overall tendencies of the rods in the image.


### Required Python Libraries
The most current version of the code was developed using the following libraries and versions:

* **numpy**  1.91
* **scipy** 0.14.0
* **matplotlib** 1.4.2
* **pillow** 2.6.0

### Using Rectangle_fit.py
There are two main programs that can be used:
* rectangle_fit.rectangle_fit is used for analyzing individual images
* rectangle_fit.fit_img_stack is used for analyzing a .tiff stack of images

###### Input:
rectangle_fit.rectangle_fit(imgName,[filename = STR,minSize = INT, threshold = INT, min_asp_mult = FLT, max_asp_mult = FLT, len_mult = FLT, area_mult=FLT, max_qual = FLT, debug = BOOL, split = BOOL, out = BOOL, optim = BOOL, display = BOOL, if_arr = BOOL])


* imgName - NECESSARY! name of the image to be processed, this image should have white featureson a black background. IF if_arr IS True THEN THE PROGRAM WILL READ AN numPy ARRAY! The intensity of the features must be consisted across the image, if this is not the case, it is recommend that a contrast normalization procedure is used.(See Image.open documentation for accepted image formats)
*   filename - name of the file to save to. (DEFAULT = imgName, entering "NONE" or not specifiedas filename sets to default.)
*   scale - a string containing the size of each pixel with the last two entries in the string beingthe units. (DEFAULT = 1px)
*   minSize - minimum size in pixels for a feature to be fitted. (DEFAULT = 10)
*   threshold - Cutoff intensity used to generate binary image for region labelling and rectanglefitting. (DEFAULT = Calculated using image and minSize)
*   min_asp_mult - multiplier for the median aspect ratio of the rectangles to get the minimumaspect ratio for the one rod features. (DEFAULT = 0.5)
*   max_asp_mult - multiplier for the median aspect ratio of the rectangles to get the maximumaspect ratio for the one rod features. (DEFAULT = 1.7)
*   len_mult - multiplier for the median length of a one rod feature, used to classify two rods features. (DEFAULT = 1.4)
*   area_mult - multiplier for the median area of the rectangles to get the maximum area for theone rod features. (DEFAULT = 2.4)
*   max_qual - maximum quality for the one rod features. (DEFAULT = 1.3)
*   split - toggles splitting of the end to end multi rod features. (DEFAULT = True)
*   out - toggles the output of data files and images from the fitting. (DEFAULT = True)
*   debug - Toggles printing of debug information. (DEFAULT = False)
*   optim - Toggles profile optimization of the rectangles. (DEFAULT = True)
*   display - Toggles display of graphs (DEFAULT = False)
*   if_arr - Toggles image imput type from string input to numPy array input. (DEFAULT = False)

###### Output:
*   "filename"_metadata.dat  - .dat file containing the metadata for the rectangles in a human friendly format.
*   "filename"_featuredata.csv - .csv file containing values for every features detected for further analysis outside of this program.
*   "filename"_rectangle_fit.png - an image of the original image with the fitted, classified rectangles superimposed on it.
                           CYAN - one rod features
                           BLUE - small features
                           YELLOW - two rod features
                           RED - three or more rod features
*   "filename"_rectangle_fit_lines.png - The original image with lines representing the major axis of the fitted rectangles in the case of one-rod features and rectangles in the case of multi-rod features.
                           CYAN - one rod features
                           BLUE - small features
                           YELLOW - two rod features
                           RED - three or more rod features
*   "filename"_rectangle_fit_optim.png - The original image with lines representing the major axis of the optimized fitted rectangles in the case of one-rod features and rectangles in the case of multi-rod features.
                           CYAN - optimized one rod features
                           BLUE - small features
                           YELLOW - two rod features
                           RED - three or more rod features
*    List of Dictionaries - Each particle has an associated dictionary containing information about the particle (area, aspect ratio, length, etc.)

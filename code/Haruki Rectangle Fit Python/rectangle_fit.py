# rectangle_fit
# 2014 - Haruki Hirasawa, Dept. of Physics and Physical Oceanography, Memorial University of Newfoundland
#        Under the supervision of Dr. Anand Yethiraj.
#        Using Algorithm from "Finding best-fitted rectangle for regions using a bisection method"
#            D. Chaudhuri, Machine Vision and Applications (2012) - find_uper and find_properties
# Version 0.2 - July 8, 2014.
#   	- Jul 8,2014 -> added profile optimization
#   	- Jul 9,2014 -> added image stack processing (fit_img_stack)
#	- November, 2014 -> Cleaned up code, changed graph and image outputs.
from PIL import Image, ImageDraw, ImageSequence
import sys
import numpy as np
import argparse
from matplotlib import pyplot as plt
import scipy 
from scipy import ndimage
import math
import time

#------------------------------------------------------------------------------------
# find_thresh 
#   Find the threshold for which the number of regions found by LABEL is highest.
# PROCEDURE:
#   Input image. Vary threshold value and use label region. Then count the number of regions.
#   Find the max value for this count, then return the threshold. The threshold is varied between a minimum value = 10
#   and a maximum value = 250
#
# NOTE:
#   For images in which the objects are not well separated, low thresholds result in many joined regions
#   and high threshold result in many objects not begin detected. Thus the maxima is in the middle
#   For images with well separated objects, low threshold do not result in joining of regions,
#   so the maxima is at low thresholds. It may be best to determine manually the best threshold in such an image,
#   as the program will pick up any possible objects, even noise or out of focus objects.
#
# INPUT:
#   	img - a 8bit, grayscale numPy image array.
#   	interval - the interval at which the threshold is tested. 
#              	DEFAULT = 5
#   	minThresh - the minimum threshold tested
#               DEFAULT = 10
#   	minSize - the minimum size of the regions
#             	DEFAULT = 10
#	display - Toggles the output of the threshold vs. regioncount graph.
#		DEFAULT = 0 (OFF)
#	filename - A string used to save the threshold vs. regioncount graph, if needed.
#		DEFAULT = ''	
#	
#           
# OUTPUT:
#   	- Returns a threshold value.
#	- If diplay == 1 then the threshold vs. regioncount graph is saved as "filename"_find_thresh.png

def find_thresh(imgArr, interval = 5, minThresh = 50, minSize=10,display=0,filename=''):
    
    regionCount = list()
    dimens = imgArr.shape
    w = dimens[0]
    h = dimens[1]
    
    for i in range(minThresh,250,interval):
        # Create a binary thresholded image for a given threshold
        binArr = np.zeros((w,h))
                
        binArr[imgArr > i] = 255

        # Label regions
        labelArr, numFeats = ndimage.label(binArr)
        if numFeats > 0:
            # Create a histogram where each bin is a region
            histo = np.histogram(labelArr,bins = numFeats)
            histo = histo[0]
            histo = histo[histo >= minSize] # count the number of bins with more than the minimum number of pixels
            regionCount.append(len(histo))
        else: regionCount.append(0)
    if display == 1:
	plt.figure(3)
	plt.plot(regionCount)
	plt.xlabel('Threshold')
	plt.ylabel('Number of Regions')
	plt.savefig(filename+"_find_thresh.png")
	plt.clf()
	plt.cla()
    bestThresh = regionCount.index(np.amax(regionCount)) # Take the maximum region count.
    return bestThresh*interval+minThresh

#------------------------------------------------------------------------------------
# FIND_PROPERTIES
# PURPOSE:
#   Calculates and returns the centroid and the angle of the major axis  from the horizontal
#   of a given binary image region.
#
# INPUT:
#   Binary image numPy array where the region in question is marked with values greater than zero.
#
# OUTPUT:
#   Three element list containing x coordinate of the centroid, y coordinate of the centroid
#   and the angle of the major axis.

def find_properties(binArr):
    # NOTE: return y-coordinates in first row, x-coords in second
    reg_coords = np.where(binArr > 0)
    xcoords = reg_coords[1]
    ycoords = reg_coords[0]

    # Calculate centroid
    xcent = np.sum(xcoords)*1.0/xcoords.size 
    ycent = np.sum(ycoords)*1.0/ycoords.size

    # Calculate major axis angle
    sum_prods = 2*np.sum((xcoords-xcent)*(ycoords-ycent))
    sum_diff_sqrs = np.sum((xcoords-xcent)**2-(ycoords-ycent)**2)
    if sum_diff_sqrs != 0:
        theta_maj = 0.5*math.atan2(sum_prods,sum_diff_sqrs)
    else: theta_maj = math.pi*0.5 # if dividing by zero, the major axis is vertical.
    return [ycent,xcent,theta_maj]

#------------------------------------------------------------------------------------
# LINE_INTERSECT
#
# PURPOSE:
#	Use a point and slope for two lines and find the point where they intersect.
#	Used to find the vertices of the rectangles.
#
# INPUT:
#	pnt1 - tuple containing the coordinates of a points on the first line.
#	pnt2 - tuple containing the coordinates of a points on the second line.
#	m1 - the slope of the first line
#	m2 - the slope of the second line
#
# OUTPUT:
#	A tuple containing the (x,y) coordinates of the intersection point.

def line_intersect(pnt1,pnt2,m1,m2):
	return (((pnt1[1]-pnt2[1])-(pnt1[0]*m1-pnt2[0]*m2))/(m2-m1),(m2*(pnt1[1]-m1*pnt1[0])-m1*(pnt2[1]-m2*pnt2[0]))/(m2-m1))

#------------------------------------------------------------------------------------
# FIND_UPER
#
# PURPOSE:
#   Finds the bounding rectangle for a region.
#
# INPUT:
#   edge - numPy array of coordinates defining the edge of a region.
#   reg_props - list of region properies [centroid x coordinate, centroid y coordinate, major axis angle]
#
# OUTPUT:
#   list of coordinates for the vertices of the bounding rectangle.
#
# PROCEDURE:
#   Described in depth by D. Chaudhuri (2012).
#   - Finds points on the edge that are furthest from the major and minor axes both above and
#       below them.
#   - Use these points and the slope of the major axis to define the vertices of the rectangle.
#

def find_uper(edge,reg_props):
	xcent = reg_props[1] #centroid coordinates
	ycent = reg_props[0] 
	m = math.tan(reg_props[2]) #slope of major axis

	xdiff = edge[:,0] - xcent #distance from centroid for edge positions
	ydiff = edge[:,1] - ycent

	if m == 0: m+= 1e-16 # need to find a better way to handle 0 slope cases
	# Find the y-intercepts for the major and minor axes.
	maj_int = ycent - xcent*m
	min_int = ycent + xcent/m
	
	# Check if the points on the edge are above/below the major/minor axes
	check_maj = (xcent*(edge[:,1]-maj_int)-(ycent-maj_int)*(edge[:,0]))
	check_min = (xcent*(edge[:,1]-min_int)-(ycent-min_int)*(edge[:,0]))

	# points above and below the major/minor axes
	abv_maj = np.where(check_maj >= 0)[0]
	blw_maj = np.where(check_maj < 0)[0]

	abv_min = np.where(check_min >= 0)[0]
	blw_min = np.where(check_min < 0)[0]
	if m != 0:
        	maj_lin = [m,maj_int]
        	min_lin = [-1/m,min_int]

        	dist2m/-aj = list()
        	dist2min = list()
		# Compute distance from the major/minor axis for each point along the edge
        	for i in range(edge.size/2):
            		dist2maj.append(abs(edge[i,1]-maj_lin[0]*edge[i,0]-maj_lin[1])/math.sqrt(maj_lin[0]*maj_lin[0]+1))
        	for i in range(edge.size/2):
            		dist2min.append(abs(edge[i,1]-min_lin[0]*edge[i,0]-min_lin[1])/math.sqrt(min_lin[0]*min_lin[0]+1))
		
		# Convert lists to numpy arrays
        	dist2maj = np.array(dist2maj)
        	dist2min = np.array(dist2min)

		# Separate the points above/below the major/minor axes.
        	dist_abv_maj = dist2maj[abv_maj]
        	dist_blw_maj = dist2maj[blw_maj]

        	dist_abv_min = dist2min[abv_min]
        	dist_blw_min = dist2min[blw_min]

		# Compute the rectangle's vertices
        	if len(dist_abv_maj) > 0 and len(dist_blw_maj) > 0 and len(dist_abv_min) > 0 and len(dist_blw_min) > 0:
            		max_abv_maj = abv_maj[np.argmax(dist_abv_maj)]
            		max_blw_maj = blw_maj[np.argmax(dist_blw_maj)]
            		max_abv_min = abv_min[np.argmax(dist_abv_min)]
            		max_blw_min = blw_min[np.argmax(dist_blw_min)]
            
            		max_abv_maj = edge[max_abv_maj,:]
            		max_blw_maj = edge[max_blw_maj,:]
            		max_abv_min = edge[max_abv_min,:]
            		max_blw_min = edge[max_blw_min,:]
            
            		top_left = line_intersect(max_abv_maj,max_abv_min,m,-1/m)
            		top_right = line_intersect(max_abv_maj,max_blw_min,m,-1/m)
            		bot_left = line_intersect(max_blw_maj,max_abv_min,m,-1/m)
            		bot_right = line_intersect(max_blw_maj,max_blw_min,m,-1/m)

            		return [top_left,top_right,bot_right,bot_left]
        	else: return [(-1,-1),(-1,-1),(-1,-1),(-1,-1)]
    	else: return [(-1,-1),(-1,-1),(-1,-1),(-1,-1)]

#------------------------------------------------------------------------------------
# END_POINT
# PURPOSE:
#   Finds the points on the edge of a region furthest from the centroid as well as the maximum 
#   distance along the edge between two end points.
#
# INPUT:
#   coords - a numPy array coordinates for the path around the edge of the region
#   reg_prop - [centroid x coordinates, centroid y coordinates, major axis angle]
#   endcheck_rad - the radius of the section over which the edge is checked for radius maxima.
#       DEFAULT = 3
#   radthresh - minimum threshold for a radius maximum. DEFAULT - 0
#   min_endpnt_sep - minimum separation between endpoints. DEFAULT = 7
#
# OUTPUT:
#    a two element list containing the maximum edge length and the number of end points.
#
# PROCEDURE:
#   - Calculate the radius from the centroid
#   - find the local maxima
#   - Remove any local maxima that are too close to one another
#   - Find the distances between each set of end points
#
def end_point(coords,reg_props,endcheck_rad = 3,radthresh=0,min_endpnt_sep = 7):
    xcent = reg_props[1] # Store coordinates of the centroid
    ycent = reg_props[0]

    endpoint = list()
    radius = ((coords[:,1]-ycent)**2+(coords[:,0]-xcent)**2)**(0.5)
        # Creates a numPy array of the distance from the centroid for each point on the edge.
    radius = radius.tolist() # Convert to list
    if coords.size/2 > endcheck_rad*2+1:
        for i in range(0,coords.size/2):
            # If at beginning, loop around to end of list
            if i -endcheck_rad < 0:
                # Find local maxima in a section around the chosen point
                loc_max = max(radius[i-endcheck_rad:coords.size/2-1]+radius[0:i+endcheck_rad+1])
                # if the local maxima is the chosen point add it to the endpoint list.
                if radius[i] == loc_max and radius[i] > radthresh:
                    endpoint.append(i)
            elif i + endcheck_rad+1 < coords.size/2:
                loc_max = max(radius[i-endcheck_rad:i+endcheck_rad+1])
                if radius[i] == loc_max and radius[i] > radthresh:
                    endpoint.append(i)            
            # if at end, loop around to beginning of list.
            else:
                loc_max = max(radius[i-endcheck_rad:coords.size/2+10]+radius[0:endcheck_rad+i-coords.size/2+1])
                if radius[i] == loc_max and radius[i] > radthresh:
                    endpoint.append(i)
        if len(endpoint) >1: # remove any endpoints that are too close by removing the one with
                             # the smaller distance to the centroid.
            i = 0
            while i+1 < len(endpoint):
                tempdist = endpoint[i] - endpoint[i+1]
                if abs(tempdist) < min_endpnt_sep:
                    if tempdist >= 0:
                        endpoint.remove(endpoint[i+1])
                    else:
                        endpoint.remove(endpoint[i])
                    i=0
                else: i+=1
                
        n_endpnts = len(endpoint)
        sect_len = list() # list of length along the edge in between each pair of adjacent endpoints.
        coords = coords.tolist() 
        for i in range(n_endpnts):
            if i == n_endpnts-1: # define the section of the edge coordinates between the end points
                section = coords[endpoint[i]:len(coords)]+coords[0:endpoint[0]]
            else: section = coords[endpoint[i]:endpoint[i+1]]
            tmp_sect_len = 0
            for j in range(len(section)-1): # find the length of the section.
                tmp_sect_len += math.sqrt((section[j+1][0]-section[j][0])**2+(section[j+1][1]-section[j][1])**2)
            sect_len.append(tmp_sect_len)
        return [max(sect_len),n_endpnts]
    else: return [0,0]
#------------------------------------------------------------------------------------
# RECTANGLE_PROPERTIES
# PURPOSE: 
#   Find various values of interest for the rectangle
#
# INPUT: 
#   A list of the coordinates of the vertices of the rectangle.
#
# OUTPUT:
#   a list containing:
#       [length,width,area, aspect ratio, cos^2(phi),centroid x coordinate, centroid y coordinate]
#       Note: phi is the angle from the horizontal.
# 

def rectangle_properties(vertices):
    # lengths of the sides
    side1 = math.sqrt((vertices[0][0]-vertices[1][0])**2+(vertices[0][1]-vertices[1][1])**2)
    side2 = math.sqrt((vertices[0][0]-vertices[3][0])**2+(vertices[0][1]-vertices[3][1])**2)
    
    if side2 !=0 and side1 != 0:
        if side1 > side2:
            if vertices[0][0]-vertices[1][0] == 0: 
		cos2phi = 0
		phi = math.pi/2
            else: 
		cos2phi = 1/(1+(vertices[0][1]-vertices[1][1])**2/(vertices[0][0]-vertices[1][0])**2)
		phi = math.atan((vertices[0][1]-vertices[1][1])/(vertices[0][0]-vertices[1][0]))
            length = side1
            width = side2
            aspect_rat = length/width
        else:
            if vertices[0][0]-vertices[3][0] == 0:
		cos2phi = 0
		phi = math.pi/2
            else: 
		cos2phi = 1/(1+(vertices[0][1]-vertices[3][1])**2/(vertices[0][0]-vertices[3][0])**2)
		phi = math.atan((vertices[0][1]-vertices[3][1])/(vertices[0][0]-vertices[3][0]))
            length = side2
            width = side1
            aspect_rat = length/width
    else:
        cos2phi = -1.0
        length = -1.0
        width = -1.0
        aspect_rat = -1.0
	phi = -1.0

    area = side1*side2
    xcent = (vertices[0][0]+vertices[2][0])/2 # Centroid calculated by taking the averages of opposite corners
    ycent = (vertices[0][1]+vertices[2][1])/2
    return [length, width, area, aspect_rat,cos2phi,xcent,ycent,phi]

#------------------------------------------------------------------------------------
# GET_PROFILE
# PURPOSE:
#    Takes two points in an image and returns the profile.
#
# Source:
# http://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array
#
# INPUT:
#	img_arr - numpy image array for the image being processed
#	points - A list containing two tuples, the two endpoints of the profile
#
# OUTPUT:
#	num - number of elements in the profile
#	prof - an numpy array containing the intensities of the pixels along the profile line
#	x - the x-coordinates of the points on the profile line
#	y - the y-coordinates of the points on the profile line

def get_profile(img_arr,points):
	num = int(math.sqrt((points[0][0]-points[1][0])**2+(points[0][1]-points[1][1])**2))
        x = np.linspace(points[0][0],points[1][0],num)
        y = np.linspace(points[0][1],points[1][1],num)
        prof = img_arr[y.astype('int'),x.astype('int')]
	return num,prof,x,y


#------------------------------------------------------------------------------------
# PROFILE_SPLIT
# PURPOSE:
#   Takes the profile of a rectangle that is fitted to multiple rods connected end to end
#   and splits the rectangle at minima on the profile.
# 
# INPUT:
#   img_arr - greyscale numPy image array.
#   rectangle - list of the coordinates of the vertices of the rectangle
#   threshfrac - determines the threshold for a minima based on a fraction of the maximum intensity
#       on the profile
#   minrad - radius of the section over which the minima are checked
#   rod_len - length of the rod, determines how close the split points are allowed to be.
#
# OUTPUT:
#   A list of lists of coordinates of the vertices of the new rectangles.
#
# PROCEDURE:
#   - Find end points for the profile
#   - Take the profile along the line between the points
#   - Find the local minima on the profile that are not on the ends and are lower than the threshold
#   - Remove the split points that are too close to one another
#   - For each split point, define two points on the rectangle's sides and define new rectangles
#   

def profile_split(img_arr,rectangle,threshfrac = 0.4,minrad = 2,rod_len = 10):
	# Define points for profile
	if math.sqrt((rectangle[0][0]-rectangle[1][0])**2+(rectangle[0][1]-rectangle[1][1])**2) <= math.sqrt((rectangle[0][0]-rectangle[3][0])**2+(rectangle[0][1]-rectangle[3][1])**2):
		pnt1 = [(rectangle[0][0]+rectangle[1][0])/2,(rectangle[0][1]+rectangle[1][1])/2]
		pnt2 = [(rectangle[2][0]+rectangle[3][0])/2,(rectangle[2][1]+rectangle[3][1])/2]
		vert = True # Used for defining new rectangle vertices.
	else:
        	pnt1 = [(rectangle[0][0]+rectangle[3][0])/2,(rectangle[0][1]+rectangle[3][1])/2]
        	pnt2 = [(rectangle[2][0]+rectangle[1][0])/2,(rectangle[2][1]+rectangle[1][1])/2]
        	vert = False
    	
 	num,prof,x,y = get_profile(img_arr,(pnt1,pnt2))
    
	thresh = threshfrac*max(prof)

	minpos = list() #list of local minima
	# Find local minima
	for i in range(int(num/4),int(3*num/4)):
        	minima = min(prof[i-minrad:i+minrad])
        	if minima == prof[i] and prof[i] < thresh: minpos.append(i)
	if (pnt2[0]-pnt1[0]) == 0:
        	m = (pnt2[1]-pnt1[1])/1e-16
    	else: m = (pnt2[1]-pnt1[1])/(pnt2[0]-pnt1[0]) #slope of the major axis
    
    	new_rect = list() #list of new rectangle vertices.
	# Remove split points that are too close to one another.
    	if len(minpos) > 0:
        	if len(minpos) >1:
            		i = 0
            		while i+1 < len(minpos):
                		tempdist = minpos[i] - minpos[i+1]
                		if abs(tempdist) < rod_len:
                    			if tempdist >= 0:
                        			minpos.remove(minpos[i+1])
                    			else:
                        			minpos.remove(minpos[i])
                    			i=0
                		else: i+=1
                    
		# Find coordinates for split points
		for j in range(len(minpos)):
		    	interv_x = (pnt2[0]-pnt1[0])/num
		    	interv_y = (pnt2[1]-pnt1[1])/num
		    	split_coords = [minpos[j]*interv_x+pnt1[0],minpos[j]*interv_y+pnt1[1]]
		    
		    	splitpnt1 = line_intersect(split_coords,rectangle[0],-1/m,m)
			splitpnt2 = line_intersect(split_coords,rectangle[2],-1/m,m)

		    # Define new rectangles and add to the list.
		    	if j == 0:
		        	if vert:
		            		new_rect.append([rectangle[0],rectangle[1],splitpnt2,splitpnt1])
		        	else:
		            		new_rect.append([rectangle[0],splitpnt1,splitpnt2,rectangle[3]])
		    	else:
		        	if vert:
		            		new_rect.append([new_rect[i-1][2],new_rect[i-1][3],splitpnt1,splitpnt2])
		        	else:
		            		new_rect.append([new_rect[i-1][1],splitpnt1,splitpnt2,new_rect[i-1][2]])
		    	if j == len(minpos)-1:
		        	if vert:
		            		new_rect.append([splitpnt1,splitpnt2,rectangle[2],rectangle[3]])
		        	else:
		            		new_rect.append([splitpnt1,rectangle[1],rectangle[2],splitpnt2])
    	else: new_rect.append(rectangle)
    
    	return new_rect


#------------------------------------------------------------------------------------
# PROFILE_CHECK_OUTOFBNDS
# PURPOSE:
#	checks to see if the line from centroid to centroid+delta extends outside the bounds
#	of the image. If it does, return a fraction to determine where to cut off the line.
#
# INPUT:
#	cent - a tuple containing the (x,y) coordinates of the starting points of the profile line
#	dimens - a tuple containing the width and height of the image in question
#	delta - a tuple containing the (dx,dy) vector which determines the end of the profile line
#
# OUTPUT:
#	f - a number between 0 and 1, indicates the fraction of the length of the line to keep.
def profile_check_outofbnds(cent,dimens,delta):
	w,h = dimens
	dx,dy = delta
	f = 1
	if cent[0]+dx > w-1: f = abs((w-1-cent[0])/dx)
	if cent[1]+dy > h-1: 
		if f == 1: f = abs((h-1-cent[1])/dy)
		else: f = min([f,abs((h-1-cent[1])/dy)])
	if cent[0]+dx < 1: f = abs((cent[0])/dx)
	if cent[1]+dy < 1: 
		if f == 1: f = abs((cent[1])/dy)
		else: f = min([f,abs((cent[1])/dy)])

	return f

#------------------------------------------------------------------------------------
# PROFILE_OPTIM
# PURPOSE:
#   Uses the profile along the axes of a rectangle to find new, more suitable bounds for the
#   rectangle.
#
# INPUT:
#   img_arr - numPy array for the image from which the profile will be taken.
#   rectangle - Coordinates of the vertices of the rectangle
#   threshfrac - fraction of the maximum intensity used to determine the new bounds.
#
# OUTPUT:
#   A 2x4 list of coordinates defining the new rectangle.
#   
# PROCEDURE:
#   - Choose end points of the two perdendicular profiles that each bisect two opposite sides of
#       the rectangle.
#   - Adjust end points so that they do not try to take the profile of points outside of the array.
#   - Take the profiles
#   - Find the first point at which the intensity drops below the threshold bot above and below
#       the center of the profile, moving outwards.
#   - Define new vertices using the cutoff points found.
#
def profile_optim(img_arr,rectangle,threshfrac=0.8):

	dimens = img_arr.shape #image dimensions

	# centroid x,y coordinate
	cent = (float(rectangle[0][0]+rectangle[2][0])/2,float(rectangle[0][1]+rectangle[2][1])/2) 
	
	# "displacement vectors" that are added/subtracted from the centroid to determine the 
	#	line along which the profile is taken. Since the vectors are the length/width of the
	#	rectangles we therefore check a profile that is twice the length/width of the rectangle.
	dr_h = (rectangle[2][0] - rectangle[1][0],rectangle[2][1] - rectangle[1][1])
	dr_v = (rectangle[1][0] - rectangle[0][0],rectangle[1][1] - rectangle[0][1])

	# Check to see if the defined profile lines are outside the bounds of the image.
	# 	If they are, return a fraction of the length of the original line which defines
	#	the new, in bounds line.
	fh2 = profile_check_outofbnds(cent,dimens,dr_h)
	fh1 = profile_check_outofbnds(cent,dimens,map(lambda dr_h:dr_h*-1,dr_h))

	fv2 = profile_check_outofbnds(cent,dimens,dr_v)
	fv1 = profile_check_outofbnds(cent,dimens,map(lambda dr_v:dr_v*-1,dr_v))

	horizpnt1 = [cent[0]-dr_h[0]*fh1,cent[1]-dr_h[1]*fh1] # Horizontal Profile end points
	horizpnt2 = [cent[0]+dr_h[0]*fh2,cent[1]+dr_h[1]*fh2] 

	vertpnt1 = [cent[0]-dr_v[0]*fv1,cent[1]-dr_v[1]*fv1] # Vertical Profile end points
	vertpnt2 = [cent[0]+dr_v[0]*fv2,cent[1]+dr_v[1]*fv2]

    # Take profile using end points found above.
    	try: 
		num_vert,vertprof, vert_x,vert_y = get_profile(img_arr,(vertpnt1,vertpnt2))
		
		num_horiz,horizprof,horiz_x,horiz_y= get_profile(img_arr,(horizpnt1,horizpnt2))        
		
		# Compute Threshold = MAX{threshfrac*(Overall max intensity),1.2*(Overall min intensity)}
		thresh = max([threshfrac*max([max(vertprof),max(horizprof)]) ,min([min(vertprof),min(horizprof)])*1.2])
		blw_thresh_vert = np.array(np.where(vertprof < thresh))

		# Find bound points
		vert_gt_cent = blw_thresh_vert[blw_thresh_vert >= 2*num_vert/3]
		if len(vert_gt_cent) > 0:vert_bndpnt1 = np.amin(vert_gt_cent)
		else: vert_bndpnt1 = num_vert-1#vert_bndpnt1 = np.argmin(vertprof[2*num_vert/3:])+num_vert/2

		vert_lt_cent = blw_thresh_vert[blw_thresh_vert < num_vert/3]
		if len(vert_lt_cent) > 0: vert_bndpnt2 = np.amax(vert_lt_cent)
		else: vert_bndpnt2=0#vert_bndpnt2 = np.argmin(vertprof[:num_vert/3])

		blw_thresh_horiz = np.array(np.where(horizprof < thresh))

		horiz_gt_cent = blw_thresh_horiz[blw_thresh_horiz >= 2*num_horiz/3]
		if len(horiz_gt_cent) > 0: horiz_bndpnt1 = np.amin(horiz_gt_cent)
		else:horiz_bndpnt1 = num_horiz-1 #horiz_bndpnt1 = np.argmin(horizprof[2*num_horiz/3:])+num_horiz/2

		horiz_lt_cent = blw_thresh_horiz[blw_thresh_horiz < num_horiz/3]
		if len(horiz_lt_cent) > 0: horiz_bndpnt2 = np.amax(horiz_lt_cent)
		else:horiz_bndpnt2=0 #horiz_bndpnt2 = np.argmin(horizprof[:num_horiz/3])

		# Find coordinates of the bound points
		v_bndpnt_coords1 = [vert_x[vert_bndpnt1],vert_y[vert_bndpnt1]]
		v_bndpnt_coords2 = [vert_x[vert_bndpnt2],vert_y[vert_bndpnt2]]

		h_bndpnt_coords1 = [horiz_x[horiz_bndpnt1],horiz_y[horiz_bndpnt1]]
		h_bndpnt_coords2 = [horiz_x[horiz_bndpnt2],horiz_y[horiz_bndpnt2]]

		# Find vertices of the new rectangle.
		if rectangle[1][0] - rectangle[2][0] == 0:
			top_left = [h_bndpnt_coords2[0],v_bndpnt_coords1[1]]
			top_right = [h_bndpnt_coords1[0],v_bndpnt_coords1[1]]
			bot_left = [h_bndpnt_coords2[0],v_bndpnt_coords2[1]]
			bot_right = [h_bndpnt_coords1[0],v_bndpnt_coords2[1]]
		else:
			if (v_bndpnt_coords2[0]-v_bndpnt_coords1[0]) == 0:
				m = (v_bndpnt_coords2[1]-v_bndpnt_coords1[1])/(1e-16)
			else: m = (v_bndpnt_coords2[1]-v_bndpnt_coords1[1])/(v_bndpnt_coords2[0]-v_bndpnt_coords1[0])
			top_left = line_intersect(h_bndpnt_coords2,v_bndpnt_coords1,m,-1/m)
			top_right = line_intersect(h_bndpnt_coords2,v_bndpnt_coords2,m,-1/m)
			bot_left = line_intersect(h_bndpnt_coords1,v_bndpnt_coords1,m,-1/m)
			bot_right = line_intersect(h_bndpnt_coords1,v_bndpnt_coords2,m,-1/m)
		return [top_left,top_right,bot_right,bot_left]
	except Exception:
		num_vert,vertprof = get_profile(img_arr,(vertpnt1,vertpnt2))
		print "err"
		return [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]

#------------------------------------------------------------------------------------
# find_RECT_AXIS
# Computes and returns the end points of the major axis of a rectangle whose vertices are given.
# Used for drawing the major axes of the rectangles onto the image.
def find_rect_axis(rectangle):
   l1 = (rectangle[0][0]-rectangle[1][0])**2+(rectangle[0][1]-rectangle[1][1])**2
   l2 = (rectangle[1][0]-rectangle[2][0])**2+(rectangle[1][1]-rectangle[2][1])**2
   if l1 > l2:
	return [(rectangle[0][0]+rectangle[3][0])/2,(rectangle[0][1]+rectangle[3][1])/2,(rectangle[1][0]+rectangle[2][0])/2,(rectangle[1][1]+rectangle[2][1])/2]
   elif l1 <= l2:
	return [(rectangle[0][0]+rectangle[1][0])/2,(rectangle[0][1]+rectangle[1][1])/2,(rectangle[3][0]+rectangle[2][0])/2,(rectangle[3][1]+rectangle[2][1])/2]

#------------------------------------------------------------------------------------
# rectangle_fit - v0 July 7, 2014
#
# PURPOSE:  
#   Takes an 8bit image of objects (specifically rod colloid images), fits rectangles to them
#   and analyses the rectangles.
#
# USAGE:
#   rectangle_fit.rectangle_fit(imgName[filename = STR,minSize = INT,threshold = INT,min_asp_mult = FLT,
#       max_asp_mult = FLT,len_mult = FLT,area_mult=FLT,max_qual = FLT, debug = BOOL,split = BOOL,
#       out = BOOL, optim = BOOL])
#
# INPUTS:
#   imgName - NECESSARY! name of the image to be processed, this image should have white features
#           on a black background. IF if_arr IS True THEN THE PROGRAM WILL READ AN numPy ARRAY!
#           The intensity of the features must be consisted across the image, if this is not the
#           case, it is recommend that a contrast normalization procedure is used.(See Image.open
#           documentation for accepted image formats)
#   filename - name of the file to save to. (DEFAULT = imgName, entering "NONE" or not specified
#           as filename sets to default.)
#   scale - a string containing the size of each pixel with the last two entries in the string being
#           the units. (DEFAULT = 1px)
#   minSize - minimum size in pixels for a feature to be fitted. (DEFAULT = 10)
#   threshold - Cutoff intensity used to generate binary image for region labelling and rectangle
#           fitting. (DEFAULT = Calculated using image and minSize)
#   min_asp_mult - multiplier for the median aspect ratio of the rectangles to get the minimum
#           aspect ratio for the one rod features. (DEFAULT = 0.5)
#   max_asp_mult - multiplier for the median aspect ratio of the rectangles to get the maximum
#           aspect ratio for the one rod features. (DEFAULT = 1.7)
#   len_mult - multiplier for the median length of a one rod feature, used to classify two rods features. (DEFAULT = 1.4)
#   area_mult - multiplier for the median area of the rectangles to get the maximum area for the
#           one rod features. (DEFAULT = 2.4)
#   max_qual - maximum quality for the one rod features. (DEFAULT = 1.3)
#   split - toggles splitting of the end to end multi rod features. (DEFAULT = True)
#   out - toggles the output of data files and images from the fitting. (DEFAULT = True)
#   debug - Toggles printing of debug information. (DEFAULT = False)
#   optim - Toggles profile optimization of the rectangles. (DEFAULT = True)
#   display - Toggles display of graphs (DEFAULT = False)
#   if_arr - Toggles image imput type from string input to numPy array input. (DEFAULT = False)
#
# OUTPUT:
#   "filename"_metadata.dat  - .dat file containing the metadata for the rectangles in a human
#                           friendly format.
#   "filename"_featuredata.csv - .csv file containing values for every features detected for
#                           further analysis outside of this program.
#   "filename"_rectangle_fit.png - an image of the original image with the fitted, classified
#                           rectangles superimposed on it.
#                           CYAN - one rod features
#                           BLUE - small features
#                           YELLOW - two rod features
#                           RED - three or more rod features
#   "filename"_rectangle_fit_lines.png - The original image with lines representing the major axis
#			    of the fitted rectangles in the case of one-rod features and rectangles
#			    in the case of multi-rod features.
#                           CYAN - one rod features
#                           BLUE - small features
#                           YELLOW - two rod features
#                           RED - three or more rod features
#   "filename"_rectangle_fit_optim.png - The original image with lines representing the major axis
#			    of the optimized fitted rectangles in the case of one-rod features and rectangles
#			    in the case of multi-rod features.
#                           CYAN - optimized one rod features
#                           BLUE - small features
#                           YELLOW - two rod features
#                           RED - three or more rod features
#    List of Dictionaries - Each particle has an associated dictionary containing information about the
#			    particle (area, aspect ratio, length, etc.)
#
#
# PROCEDURE:
#   - Image.open() reads the image and places it into an numPy array.
#   - Using the calculated or given threshold a binary image of the features is created.
#   - ndimage.label() labels the separate features and gives each a number.
#   - A rectangle is fitted over each features and values are calculated for each. 
#   - Median values for all rectangles are used to determine if any features are rods connected
#       end to end and splits those that are.
#   - Rods are classified based on various criteria
#   - Analysis information is output onto files.
#
#

def rectangle_fit(imgName,filename="NONE",scale ="1px",minSize=10,threshold=-10,min_asp_mult = 0.5,max_asp_mult = 1.7,
                  split=True,out=True,area_mult=2.4,max_qual = 1.3,debug=False,optim = True,len_mult = 1.4,
                  display = False,if_arr = False):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # INITIALIZE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Take initial time for time elapsed calculation.
    time1 = time.clock()

    if filename =="NONE":
        if if_arr: filename = "no_name_"+time.asctime()
        else: filename = imgName[0:len(imgName)-4]
    
    # Read in image into numPy Array.
    if if_arr: # IF THE INPUT IS AN ARRAY
        imgArr = imgName
        imgName = filename
	img = Image.fromarray(imgArr)
	img_rectangles = img.convert('L') 
	img_lines= img.convert('L')
	img_rectangles = img_rectangles.convert('RGB')
	img_lines = img_lines.convert('RGB')
    else:
        img = Image.open(imgName) # open image
        imgArr = np.array(img) # place into a numPy array
	img = img.convert('L') # We want to convert to grayscale so we convert to 8-bit
	img_rectangles = img.convert('RGB') # We then want draw RGB lines on the images so we convert to RGB
	img_lines = img.convert('RGB')
	img_optim =img.convert('RGB')
	

    # Set threshold if not specified.
    if threshold == -10: threshold = find_thresh(imgArr,minSize=minSize,display=display,filename =filename)

    # Plot image for use in output.
    plt.figure(1)
    plt.clf()
    imgplot = plt.imshow(imgArr)
    imgplot.set_cmap('gray')
    plt.title('Fitted Rectangles')
    plt.text(10,30,"1px = "+scale,color="red")
    
    # Take dimensions
    dimens = imgArr.shape
    w = dimens[0] # width
    h = dimens[1] # height 

    print 'width: ', w
    print 'height: ', h
    print 'threshold: ', threshold

    binImgArr = np.where(imgArr > threshold, 1,0) # Thresholded binary image array
    labelArr, numFeats = ndimage.label(binImgArr) # Array of Labeled regions

    print 'Number of features: ', numFeats

    # Create lists for values:

    data = list() # data will be a list of dictionaries

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # FITTING
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # initial fitting
    print "Fitting Rectangles..."
    for i in range(numFeats):

        if i % (int(numFeats/10)) == 0: print "Feature: ",i,"/",numFeats
        # Create an array containing the feature in question
        temp_img = np.where(labelArr == i+1,1,0)
        if temp_img[temp_img==1].size > minSize:
            # Calculates centroid and angle of major axis of the feature.
            reg_props = find_properties(temp_img)
	    
            # Finds the coordinates of the path of the contour around feature.
            contour = plt.contour(temp_img,[0.0]) # find and plot contours
            path =  contour.allsegs[0] # find contour paths
            if len(path) > 1: # Find the index of the longest path
                pathlen = list()
                for i in range(len(path)): pathlen.append(len(path[i]))
                maxpath = max(pathlen)
                path_ind = pathlen.index(maxpath)
            else: path_ind = 0
            path = path[path_ind] 
            coords = np.array(path) # Convert path to numPy array
        
            # Remove duplicate points
            i=0
            while i < coords.size/2-1:
                if coords[i][0] == coords[i+1][0] and coords[i][1] == coords[i+1][1]:
                    coords = np.delete(coords,i+1,0)
                else: i+=1
            
            if len(coords)> 0:
                # Fit Rectangles
                temprect = find_uper(coords,reg_props)
                rect_prop = rectangle_properties(temprect)
                # Get end point data from edge coordinates
                end_pnt_data = end_point(coords,reg_props)

                # Calculate quality
                reg_size = temp_img[temp_img==1].size
                qual = (rect_prop[2]-reg_size)/reg_size
		particle_data = {'length':rect_prop[0],'width':rect_prop[1],
			'area':rect_prop[2],'aspectratio':rect_prop[3],
			'cos2phi':rect_prop[4],'max_edge_len':end_pnt_data[0],
			'end_pnt':end_pnt_data[1],'rectangle':temprect,
			'xcent':reg_props[1],'ycent':reg_props[0],'phi':rect_prop[7]}
                if qual < 1000: particle_data['quality'] = qual
		data = data+[particle_data]
		
		
                
    print "Fitting Complete."

    # Calculate median values for fitted rectangles, used for classifying rectangles
    med_wid = np.median([x['width'] for x in data if 'width' in x])
    med_len = np.median([x['length'] for x in data if 'length' in x])
    med_asp_rat = np.median([x['aspectratio'] for x in data if 'aspectratio' in x])
    med_area = np.median([x['area'] for x in data if 'area' in x])

    min_asp_rat = med_asp_rat*min_asp_mult # minimum aspect ratio for an one rod feature
    max_asp_rat = med_asp_rat*max_asp_mult # maximum aspect ratio for an one rod feature

    # Initialize counters
    n1rod=0 # number of one rod features
    n2rod=0 # number of two rod features
    n3plusrod=0 # number of three plus rod features

    area_1rod = 0 # total area of one rod features
    area_multrod = 0 # total area of multi rod features
    numadded=0 # number of rods added by splitting.

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SPLITTING RECTANGLES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print "Splitting end to end rods: "
    if split:
        for d in data:
            if 'quality' in d and d['quality'] < max_qual and d['aspectratio'] > max_asp_rat:
                new_rects = profile_split(imgArr, d['rectangle'],rod_len = med_len*0.7)
                
                if len(new_rects) > 1:
                    temp_qual = d['quality']
                    
                    
                    for j in new_rects:
			props = rectangle_properties(j)
                        # Calculate and add values for the new rectangles.
			tempdict = {'rectangle':j,'quality':temp_qual,
				'length':props[0],'width':props[1],'area':props[2],
				'aspectratio':props[3],'cos2phi':props[4],
				'xcent':props[5],'ycent':props[6],'end_pnt':2,
				'max_edge_len':med_len,'phi':props[7]}
			data = data+[tempdict]
                        
                    numadded += len(new_rects)-1
		    # Delete values from original rectangle
		    del data[data.index(d)]
		
    print "Rods split: ", numadded
    numFeats += numadded # increase number of features
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CLASSIFY AND OPTIMIZE RECTANGLES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Setup Draw ojects:
    draw_img_rect = ImageDraw.Draw(img_rectangles)
    draw_img_lines = ImageDraw.Draw(img_lines)
    draw_img_optim = ImageDraw.Draw(img_optim)

    print "Optimizing Rectangles..."
    for d in data:
        if 'quality' in d:
            # Criteria for a one rod feature
            if d['quality'] < max_qual and d['aspectratio'] > min_asp_rat and d['area'] < med_area*area_mult:
                n1rod += 1 # increment one rod feature counter
                area_1rod += labelArr[labelArr == i+1].size # add area of feature to total
                d['rod_type'] = 1
		d['color'] = 'cyan'
		
                if optim:
                    optim_rect = profile_optim(imgArr,d['rectangle'])
                    optim_props=rectangle_properties(optim_rect)
		    d['optim_rect'] = optim_rect
		    d['optim_len'] = optim_props[0]
		    d['optim_wid'] = optim_props[1]
		    d['optim_cos2phi'] = optim_props[4]

		    draw_img_optim.line(find_rect_axis(optim_rect),d['color']) 

	    else:

                # Separate out features that are too small to be multi rod features
                if d['area'] < med_area*1.2:
                    area_1rod += len(labelArr[np.where(labelArr == i+1)])
                    d['color'] = 'blue'
                    d['rod_type'] = 1
                    n1rod += 1 # Count these are one rod features.
                # All other features are assumed to be multiple rods.
                else:
                    d['rod_type'] = 2
                    d['color'] = 'red'
                    area_multrod += len(labelArr[np.where(labelArr == i+1)])

    print "Optimization Complete"

    singlerod_med_len = np.median([x['length'] for x in data if ('rod_type' in x and x['rod_type'] == 1)]) # median length for single rod features only
    singlerod_med_wid = np.median([x['width'] for x in data if ('rod_type' in x and x['rod_type'] == 1)]) # median width for single rod features only
    if debug: print singlerod_med_len
    for d in data:
        if 'quality' in d and d['rod_type'] == 2 :
            # Criteria for two rod features.
            if d['end_pnt'] <= 4 and d['area'] < (singlerod_med_len*len_mult)**2 and (d['max_edge_len'] <= singlerod_med_len*len_mult*2 or (d['quality'] < max_qual and d['max_edge_len'] < singlerod_med_len*len_mult*2.5)):
                n2rod += 1
                d['color'] = 'yellow'
                d['rod_type'] = 2
            else:
                d['rod_type'] = 3
                n3plusrod += 1
                d['color'] = 'red'
        # Overlay rectangles onto the input image.
	if (out or display) and 'color' in d:
		draw_img_rect.polygon(d['rectangle'],outline=d['color'])
		
		if (d['color'] == 'cyan' or d['color'] == 'blue'):
			draw_img_lines.line(find_rect_axis(d['rectangle']),d['color'])
		else:
			draw_img_lines.polygon(d['rectangle'],outline=d['color'])
		
		if optim == 1:
			if d['rod_type'] == 2 or d['rod_type'] == 3:
				draw_img_optim.polygon(d['rectangle'],outline=d['color'])
        

    time2 = time.clock()
    
    if debug: print "Time Elapsed: ", time2 - time1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # OUTPUT
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Print to file
    if out:
        scale_num = float(scale[0:len(scale)-2])
        scale_units = scale[len(scale)-2:]
        metafile = open(filename+"_metadata.dat","w")
        featdata = open(filename+"_featuredata.csv","w")
        # Save figure as image
	if scale != "1px":
		scale_order = math.log(scale_num,10)
		
	# Save images
	img_rectangles.save(filename+"_rectangle_fit.png","PNG")
	img_lines.save(filename+"_rectangle_fit_lines.png","PNG")
	img_optim.save(filename+"_rectangle_fit_optim.png","PNG")

        # METAFILE WRITE:
        # The metafile contains overall fitting information in an easy to read format
        metafile.write(filename+"_metadata.dat"+" - "+time.strftime("%c")
                       +"\n {0:40s} {1:10f}".format("Time Elapsed (s): ",time2 - time1)
                       +"\n -------------------------------------------------------------------"
                       +"\n Image Information: "
                       +"\n {0:40s} {1:>s}".format("Image Name: ",imgName)
                       +"\n {0:40s} {1:10d}".format("Image width: ",w)
                       +"\n {0:40s} {1:10d}".format("Image height: ",h)
                       +"\n {0:40s} {1:>10s}".format("Image scale: 1px =",scale)
                       +"\n -------------------------------------------------------------------"
                       +"\n Settings: "
                       +"\n {0:40s} {1:10d}".format("Threshold: ",threshold)
                       +"\n {0:40s} {1:10d}px".format("Minimum Region Size: ",minSize)
                       +"\n {0:40s} {1:10f}".format("Minimum Aspect Ratio multiplier: ",min_asp_mult)
                       +"\n {0:40s} {1:10f}".format("Maximum Aspect Ratio multiplier: ",max_asp_mult)
                       +"\n {0:40s} {1:10f}".format("Maximum Quality: ",max_qual)
                       +"\n {0:40s} {1:10f}".format("Maximum Area multiplier: ",area_mult)
                       +"\n {0:40s} {1:10f}".format("Maximum Length multiplier: ",len_mult)
                       +"\n {0:40s} {1:>10s}".format("Splitting: ",str(split))
                       +"\n {0:40s} {1:>10s}".format("Optimization: ",str(optim))
                       +"\n -------------------------------------------------------------------"
                       +"\n Classification Data: "
                       +"\n {0:40s} {1:10d}".format("Total Features: ",numFeats)
                       +"\n {0:40s} {1:10f}".format("Number of Features Split: ",numadded)
                       +"\n {0:40s} {1:10d}".format("Total good features: ",n1rod+n2rod+n3plusrod)
                       +"\n {0:40s} {1:10d}".format("One rod features: ",n1rod)
                       +"\n {0:40s} {1:10d}".format("Two rod features: ",n2rod)
                       +"\n {0:40s} {1:10d}".format("Three plus rod features: ",n3plusrod)
                       +"\n -------------------------------------------------------------------"
                       +"\n Area fraction:"
                       +"\n {0:40s} {1:10f}px^2 = {2:10f}{3:2s}^2".format("One rod feature area: ",area_1rod,area_1rod*scale_num**2,scale_units)
                       +"\n {0:40s} {1:10d}px^2 = {2:10f}{3:2s}^2".format("Multi rod feature area: ",area_multrod,area_multrod*scale_num**2,scale_units)
                       +"\n {0:40s} {1:10f}".format("One rod feature fraction: ",((area_1rod)/(area_multrod+area_1rod))))
	d_temp = [d for d in data if 'length' in d]
        metafile.write("\n -------------------------------------------------------------------"
                       +"\n Averages (all rectangles):"
                       +"\n Length: "
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Mean: ",np.mean([d['length'] for d in d_temp]),np.mean([d['length'] for d in d_temp])*scale_num,scale_units)
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Median: ",np.median([d['length'] for d in d_temp]),np.median([d['length'] for d in d_temp])*scale_num,scale_units)
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Standard Deviation: ",np.std([d['length'] for d in d_temp]),np.std([d['length'] for d in d_temp])*scale_num,scale_units)
                       +"\n Width: "
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Mean: ",np.mean([d['width'] for d in d_temp ]),np.mean([d['width'] for d in d_temp ])*scale_num,scale_units)
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Median: ",np.median([d['width'] for d in d_temp ]),np.median([d['width'] for d in d_temp ])*scale_num,scale_units)
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Standard Deviation: ",np.std([d['width'] for d in d_temp ]),np.std([d['width'] for d in d_temp ])*scale_num,scale_units)
                       +"\n Area"
                       +"\n {0:40s} {1:10f}px^2 = {2:10f}{3:2s}^2".format("   Mean: ",np.mean([d['area'] for d in d_temp ]),np.mean([d['area'] for d in d_temp ])*scale_num**2,scale_units)
                       +"\n {0:40s} {1:10f}px^2 = {2:10f}{3:2s}^2".format("   Median: ",np.median([d['area'] for d in d_temp ]),np.median([d['area'] for d in d_temp ])*scale_num**2,scale_units)
                       +"\n {0:40s} {1:10f}px^2 = {2:10f}{3:2s}^2".format("   Standard Deviation: ",np.std([d['area'] for d in d_temp ]),np.std([d['area'] for d in d_temp ])*scale_num**2,scale_units)
                       +"\n Aspect Ratio"
                       +"\n {0:40s} {1:10f}".format("   Mean: ",np.mean([d['aspectratio'] for d in d_temp ]))
                       +"\n {0:40s} {1:10f}".format("   Median: ",np.median([d['aspectratio'] for d in d_temp ]))
                       +"\n {0:40s} {1:10f}".format("   Standard Deviation: ",np.std([d['aspectratio'] for d in d_temp ]))
                       +"\n Quality"
                       +"\n {0:40s} {1:10f}".format("   Mean: ",np.mean([d['quality'] for d in d_temp ]))
                       +"\n {0:40s} {1:10f}".format("   Median: ",np.median([d['quality'] for d in d_temp ]))
                       +"\n {0:40s} {1:10f}".format("   Standard Deviation: ",np.std([d['quality'] for d in d_temp ]))
                       +"\n Cos^2(PHI)"
                       +"\n {0:40s} {1:10f}".format("   Mean: ",np.mean([d['cos2phi'] for d in d_temp ]))
                       +"\n {0:40s} {1:10f}".format("   Median: ",np.median([d['cos2phi'] for d in d_temp ]))
                       +"\n {0:40s} {1:10f}".format("   Standard Deviation: ",np.std([d['cos2phi'] for d in d_temp ])))
	d_temp = [d for d in data if 'rod_type' in d and d['rod_type'] == 1]
        metafile.write("\n -------------------------------------------------------------------"
                       +"\n Averages (Single rod features - threshold):"
                       +"\n Length: "
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Mean: ",np.mean([d['length'] for d in d_temp]),np.mean([d['length'] for d in d_temp])*scale_num,scale_units)
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Median: ",np.median([d['length'] for d in d_temp]),np.median([d['length'] for d in d_temp])*scale_num,scale_units)
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Standard_deviation: ",np.std([d['length'] for d in d_temp]),np.std([d['length'] for d in d_temp])*scale_num,scale_units)
                       +"\n Width: "
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Mean: ",np.mean([d['width'] for d in d_temp]),np.mean([d['width'] for d in d_temp])*scale_num,scale_units)
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Median: ",np.median([d['width'] for d in d_temp]),np.median([d['width'] for d in d_temp])*scale_num,scale_units)
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Standard_deviation: ",np.std([d['width'] for d in d_temp]),np.std([d['width'] for d in d_temp])*scale_num,scale_units)
                       +"\n Area"
                       +"\n {0:40s} {1:10f}px^2 = {2:10f}{3:2s}^2".format("   Mean: ",np.mean([d['area'] for d in d_temp]),np.mean([d['area'] for d in d_temp])*scale_num**2,scale_units)
                       +"\n {0:40s} {1:10f}px^2 = {2:10f}{3:2s}^2".format("   Median: ",np.median([d['area'] for d in d_temp]),np.median([d['area'] for d in d_temp])*scale_num**2,scale_units)
                       +"\n {0:40s} {1:10f}px^2 = {2:10f}{3:2s}^2".format("   Standard_deviation: ",np.std([d['area'] for d in d_temp]),np.std([d['area'] for d in d_temp])*scale_num**2,scale_units)
                       +"\n Aspect Ratio"
                       +"\n {0:40s} {1:10f}".format("   Mean: ",np.mean([d['aspectratio'] for d in d_temp]))
                       +"\n {0:40s} {1:10f}".format("   Median: ",np.median([d['aspectratio'] for d in d_temp]))
                       +"\n {0:40s} {1:10f}".format("   Standard_deviation: ",np.std([d['aspectratio'] for d in d_temp]))
                       +"\n Quality"
                       +"\n {0:40s} {1:10f}".format("   Mean: ",np.mean([d['quality'] for d in d_temp]))
                       +"\n {0:40s} {1:10f}".format("   Median: ",np.median([d['quality'] for d in d_temp]))
                       +"\n {0:40s} {1:10f}".format("   Standard_deviation: ",np.std([d['quality'] for d in d_temp]))
                       +"\n Cos^2(PHI)"
                       +"\n {0:40s} {1:10f}".format("   Mean: ",np.mean([d['cos2phi'] for d in d_temp]))
                       +"\n {0:40s} {1:10f}".format("   Median: ",np.median([d['cos2phi'] for d in d_temp]))
                       +"\n {0:40s} {1:10f}".format("   Standard_deviation: ",np.std([d['cos2phi'] for d in d_temp])))
        if optim:
	    d_temp = [d for d in data if 'optim_len' in d]
            metafile.write(
                       "\n -------------------------------------------------------------------"
                       +"\n Averages (Single rod features - optimized):"
                       +"\n Length: "
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Mean: ",np.mean([i["optim_len"] for i in d_temp ]),np.mean([i["optim_len"] for i in d_temp ])*scale_num,scale_units)
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Median: ",np.median([[i["optim_len"] for i in d_temp ]]),np.median([[i["optim_len"] for i in d_temp ]])*scale_num,scale_units)
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Standard Deviation: ",np.std([[i["optim_len"] for i in d_temp ]]),np.std([[i["optim_len"] for i in d_temp ]])*scale_num,scale_units)
                       +"\n Width: "
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Mean: ",np.mean([i["optim_wid"] for i in d_temp ]),np.mean([i["optim_wid"] for i in d_temp ])*scale_num,scale_units)
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Median: ",np.median([i["optim_wid"] for i in d_temp ]),np.median([i["optim_wid"] for i in d_temp ])*scale_num,scale_units)
                       +"\n {0:40s} {1:10f}px = {2:10f}{3:2s}".format("   Standard Deviation: ",np.std([i["optim_wid"] for i in d_temp ]),np.std([i["optim_wid"] for i in d_temp ])*scale_num,scale_units)
                       +"\n Area"
                       +"\n {0:40s} {1:10f}px^2 = {2:10f}{3:2s}^2".format("   Mean: ",np.mean([i["optim_len"]*i["optim_wid"] for i in d_temp]),np.mean([i["optim_len"]*i["optim_wid"] for i in d_temp])*scale_num**2,scale_units)
                       +"\n {0:40s} {1:10f}px^2 = {2:10f}{3:2s}^2".format("   Median: ",np.median([i["optim_len"]*i["optim_wid"] for i in d_temp]),np.median([i["optim_len"]*i["optim_wid"] for i in d_temp])*scale_num**2,scale_units)
                       +"\n {0:40s} {1:10f}px^2 = {2:10f}{3:2s}^2".format("   Standard Deviation: ",np.std([i["optim_len"]*i["optim_wid"] for i in d_temp]),np.std([i["optim_len"]*i["optim_wid"] for i in d_temp])*scale_num**2,scale_units)
                       +"\n Aspect Ratio"
                       +"\n {0:40s} {1:10f}".format("   Mean: ",np.mean([i["optim_len"]/i["optim_wid"] for i in d_temp]))
                       +"\n {0:40s} {1:10f}".format("   Median: ",np.median([i["optim_len"]/i["optim_wid"] for i in d_temp]))
                       +"\n {0:40s} {1:10f}".format("   Standard Deviation: ",np.std([i["optim_len"]/i["optim_wid"] for i in d_temp]))
                       +"\n Cos^2(PHI)"
                       +"\n {0:40s} {1:10f}".format("   Mean: ",np.mean([i["optim_cos2phi"] for i in d_temp]))
                       +"\n {0:40s} {1:10f}".format("   Median: ",np.median([i["optim_cos2phi"] for i in d_temp]))
                       +"\n {0:40s} {1:10f}".format("   Standard Deviation: ",np.std([i["optim_cos2phi"] for i in d_temp])))
        metafile.write(
                       "\n -------------------------------------------------------------------"
                       +"\n Classification Criteria: "
                       +"\n One rod: "
                       +"\n {0:>40s} {1:<10f}".format("Quality > ",1.3)
                       +"\n {0:>40s} {1:<10f}".format("Aspect Ratio > ",min_asp_rat)
                       +"\n {0:>40s} {1:<10f}".format("Aspect Ratio > ",max_asp_rat)
                       +"\n {0:>40s} {1:<10f}px^2".format("Area < ",med_area*area_mult)
                       +"\n Small rod: "
                       +"\n {0:>40s} {1:<10f}px^2".format("Area < ",med_area*1.)
                       +"\n Two rod: "
                       +"\n {0:>40s} {1:<10f}".format("End points < ",4)
                       +"\n {0:>40s} {1:<10f}px^2".format("Area < ",(singlerod_med_len*len_mult)**2)
                       +"\n {0:>40s}".format("AND")
                       +"\n {0:>40s} {1:<10f}px".format("Max edge length < ",singlerod_med_len*len_mult*2)
                       +"\n {0:>40s}".format("OR")
                       +"\n {0:>40s} {1:<10f}".format("Quality < ",1.3)
                       +"\n {0:>40s} {1:<10f}px".format("Max edge length < ",singlerod_med_len*len_mult*2.5))
        # FEATDATA
        # a .csv file containing information for individual rods.

	if optim:
	    featdata.write("\"Feat Type\",\"Quality\",\"Width\",\"Length\",\"Area\",\"Aspect Ratio\",\"End points\",\"Max Edge Length\",\"Cos^2(PHI)\",\"Centroid (X)\",\"Centroid (Y)\",\"Optim Length\",\"Optim Width\",\"Optim Cos^2(phi)\"")
            for d in data:
		try:
            	    featdata.write("\n{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(d["rod_type"],d["quality"],d["width"],
                                                                                       d["length"],d['area'],d['aspectratio'],d['end_pnt'],
                                                                                       d['max_edge_len'],d['cos2phi'],d['xcent'],d['ycent']
                                                                                       ,d['optim_len'],d['optim_wid'],d['optim_cos2phi']))
		except: 
		    featdata.write("\n{},{},{},{},{},{},{},{},{},{},{}".format(d["rod_type"],d["quality"],d["width"],
                                                                                       d["length"],d['area'],d['aspectratio'],d['end_pnt'],
                                                                                       d['max_edge_len'],d['cos2phi'],d['xcent'],d['ycent']))

	else:
	    featdata.write("\"Feat Type\",\"Quality\",\"Width\",\"Length\",\"Area\",\"Aspect Ratio\",\"End points\",\"Max Edge Length\",\"Cos^2(PHI)\",\"Centroid (X)\",\"Centroid (Y)\"")
            for d in data:
		try:
                    featdata.write("\n{},{},{},{},{},{},{},{},{},{},{}".format(d["rod_type"],d["quality"],d["width"],
                                                                                       d["length"],d['area'],d['aspectratio'],d['end_pnt'],
                                                                                       d['max_edge_len'],d['cos2phi'],d['xcent'],d['ycent']))

        metafile.close()
        featdata.close()
    if display:
        # Plot Histogram of data for the rods.
	plt.figure(2)
        #plt.subplot(211)
        n, bins, patches = plt.hist([d['cos2phi'] for d in data if 'cos2phi' in d and d['rod_type'] == 1], bins=50, normed=1, facecolor='r')
        plt.ylabel('Number of Features')
        plt.xlabel('cos^2(PHI)')
	plt.savefig(filename+"_cos2phiHist.png")
        #plt.subplot(212)
	plt.clf()
	plt.cla()

	n, bins, patches = plt.hist([d['phi'] for d in data if 'phi' in d and d['rod_type'] == 1], bins=50, normed=1, facecolor='r')
        plt.ylabel('Number of Features')
        plt.xlabel('PHI - Angle from horizontal')
	plt.savefig(filename+"_phiHist.png")
        #plt.subplot(212)
	plt.clf()
	plt.cla()

        n, bins, patches = plt.hist([d['aspectratio'] for d in data if 'aspectratio' in d and d['rod_type'] == 1], bins=50, normed=1, facecolor='r')
        plt.ylabel('Number of Features')
        plt.xlabel('Aspect Ratio')
	plt.savefig(filename+"_aspratHist.png")
	plt.clf()
	plt.cla()

        #plt.subplot(211)
        n, bins, patches = plt.hist([d['quality'] for d in data if 'quality' in d and d['rod_type'] == 1], bins=50, normed=1, facecolor='r')
        plt.ylabel('Number of Features')
        plt.xlabel('Quality')
	plt.savefig(filename+"_qualityHist.png")
        #plt.subplot(212)
	plt.clf()
	plt.cla()

        n, bins, patches = plt.hist([d['length'] for d in data if 'length' in d and d['rod_type'] == 1], bins=50,range=(0,50), normed=1, facecolor='r')
        plt.ylabel('Number of Features')
        plt.xlabel('Length (px)')
	plt.savefig(filename+"_lengthHist.png")
        #plt.show()
	plt.clf()
	plt.cla()

    return {"threshold":threshold,"data":data}

# -------------------------------------------------------------
# FIT_IMG_STACK
#
# PURPOSE:
#   Read and process a stack of images, then prints the meta data to a .csv file.
#
# USAGE:
#   rectangle_fit.fit_img_stack(stack_name[,series_units = STR,series_vals = LIST,filename = STR,scale=STR,minSize=NUM,
#       threshold=NUM,min_asp_mult=NUM,max_asp_mult=NUM,len_mult=NUM,area_mult=NUM,max_qual=NUM,split=[TRUE|FALSE],
#       out=[TRUE|FALSE],debug=[TRUE|FALSE],optim=[TRUE|FALSE],display=[TRUE|FALSE]])
#
#   This program takes in stacks/movies of images and runs them through rectangle_fit. The program then collects
#   statistical data and prints it to a .csv file. The primary work of this program is reading, organizing, and naming
#   images from the stack using the series_units and series_vals. The default settings work of the assumption that the
#   image stack is a voltage series.
#
# INPUT:
#   stack_name - NECESSARY! name of the image stack to be processed, this image should have white features
#           on a black background. The intensity of the features must be consisted across the image, if this is not the
#           case, it is recommend that a contrast normalization procedure is used. (See Image.open
#           documentation for accepted image formats)
#   series_units - Units of the series' varying value. (DEFAULT = "V")
#   series_vals - list of values for the series, the independent variable. (DEFAULT = iterate by 10)
#   filename - name of the file to save to. (DEFAULT = imgName, entering "NONE" or not specified
#           as filename sets to default.)
#   scale - a string containing the size of each pixel with the last two entries in the string being
#           the units. (DEFAULT = 1px)
#   minSize - minimum size in pixels for a feature to be fitted. (DEFAULT = 10)
#   threshold - Cutoff intensity used to generate binary image for region labelling and rectangle
#           fitting. (DEFAULT = Calculated using image and minSize)
#   min_asp_mult - multiplier for the median aspect ratio of the rectangles to get the minimum
#           aspect ratio for the one rod features. (DEFAULT = 0.5)
#   max_asp_mult - multiplier for the median aspect ratio of the rectangles to get the maximum
#           aspect ratio for the one rod features. (DEFAULT = 1.7)
#   len_mult - multiplier for the median length of a one rod feature, used to classify two rods features. (DEFAULT = 1.4)
#   area_mult - multiplier for the median area of the rectangles to get the maximum area for the
#           one rod features. (DEFAULT = 2.4)
#   max_qual - maximum quality for the one rod features. (DEFAULT = 1.3)
#   split - toggles splitting of the end to end multi rod features. (DEFAULT = True)
#   out - toggles the output of data files and images from the fitting. (DEFAULT = True)
#   debug - Toggles printing of debug information. (DEFAULT = False)
#   optim - Toggles profile optimization of the rectangles. (DEFAULT = True)
#   display - Toggles display of graphs. (DEFAULT = False)
#
# OUTPUT:
#   A number of graphs as well as a .csv file containing metadata from the images.
#

def fit_img_stack(stack_name,scale = "1px",series_units = "V",series_vals = [-1],filename="NONE",minSize=10,
                  threshold=-10,min_asp_mult = 0.5,max_asp_mult = 1.7,split=True,out=True,
                  area_mult=2.4,max_qual = 1.3,debug=False,optim = True,len_mult = 1.4,
                  display = False):
    if filename == "NONE":
        filename = stack_name[0:len(stack_name)-4]
    
    img_stack = Image.open(stack_name)
    data = list() # A list for all the output lists
    numreg = list() # Number of features fitted
    time_elapsed = list() # Time elapsed during processing.
    index = 0
    
    # Run rectangle fitting on each image in the stack.
    for frame in ImageSequence.Iterator(img_stack):
        imgarr = np.array(frame)
        
        if series_vals[0] == -1:
            img_name = filename+"_"+str(index*10)+series_units
        else:
            img_name = filename+"_"+str(series_vals[index])+series_units
        time1 = time.clock()
        output = rectangle_fit(imgarr,filename = img_name,scale = scale,if_arr = True,minSize=minSize,threshold=threshold,
                      min_asp_mult = min_asp_mult,max_asp_mult = max_asp_mult,split=split,out=out,
                      area_mult=area_mult,max_qual = max_qual,debug=debug,optim = optim,
                      len_mult = len_mult,display = display)
        time2 = time.clock()
        data.append([output["threshold"],output["data"]])
        time_elapsed.append(time2 - time1)
        numreg.append(len(output[0]))
        index += 1

    stack_data=[]
    
    # Compute Average Values for each image
    for i in range(index):
	img_avgs ={}
	img_avgs['cos2phi_mean'] = np.mean([x["cos2phi"] for x in data[i][1] if x["rod_type"] == 1])
	img_avgs['cos2phi_median'] = np.median([x["cos2phi"] for x in data[i][1] if x["rod_type"] == 1])
	img_avgs['cos2phi_stddev'] = np.std([x["cos2phi"] for x in data[i][1] if x["rod_type"] == 1])
	
	img_avgs['asp_rat_mean'] = np.mean([x["aspectratio"] for x in data[i][1] if x["rod_type"] == 1])
	img_avgs['asp_rat_median'] = np.median([x["aspectratio"] for x in data[i][1] if x["rod_type"] == 1])	
	img_avgs['asp_rat_stddev'] = np.std([x["aspectratio"] for x in data[i][1] if x["rod_type"] == 1])
        
	img_avgs['num_1rod'] = len([1 for x in data[i][1] if x["rod_type"] == 1])
	img_avgs['num_2rod'] = len([2 for x in data[i][1] if x["rod_type"] == 2])
	img_avgs['num_3+rod'] = len([3 for x in data[i][1] if x["rod_type"] == 3])
	
	stack_data = stack_data +[img_avgs]


    if "V" in series_units: x_lab = "Voltage ("+series_units+")" # Determine the x-axis values and its units
    elif "s" in series_units: x_lab = "Time ("+series_units+")"
    elif "Hz" in series_units: x_lab = "Frequency ("+series_units+")"
    else: x_lab = series_units

    if series_vals[0] == -1: series_vals = range(0,index*10,10)

    # Print metadata to a .csv file for further processing.
    stackdata = open(filename+"_stack_data.csv","w")
    stackdata.write("\"Value\",\"Threshold\",\"Mean Cos^2(phi)\",\"Median Cos^2(phi)\",\"Standard Deviation Cos^2(phi)\",\"Mean Aspect Ratio\",\"Median Aspect Ratio\",\"Standard Deviation Aspect Ratio\",\"Number of Features\",\"Process Time\",\"Num one rod feats\",\"Num two rod feats\",\"Num three+ rod feats\"")
    for x in stack_data:
        if series_vals[0] == -1:
            stackdata.write("\n {}{},{}, {},{},{},{},{},{},{},{},{},{},{}".format(i*10,series_units,data[i][0],x["cos2phi_mean"],x["cos2phi_median"],
                                                                               x["cos2phi_stddev"],x["asp_rat_mean"],x["asp_rat_median"],
                                                                               x["asp_rat_stdev"],numreg[i],time_elapsed[i],x["num_1rod"],
                                                                               x["num_2rod"],x["num_3+rod"]))
        else:
            stackdata.write("\n {}{},{}, {},{},{},{},{},{},{},{},{},{},{}".format(series_vals[i],series_units,data[i][0],x["cos2phi_mean"],x["cos2phi_median"],
                                                                               x["cos2phi_stddev"],x["asp_rat_mean"],x["asp_rat_median"],
                                                                               x["asp_rat_stdev"],numreg[i],time_elapsed[i],x["num_1rod"],
                                                                               x["num_2rod"],x["num_3+rod"]))
            
    if display:
        stackdata.close()
        plt.figure(1)
        plt.subplot(211)
        plt.plot(series_vals,[x["cos2phi_mean"] for x in stackdata],"-bo")
        plt.ylabel('Mean Cos^2(PHI)')
        plt.xlabel(x_lab)
        plt.subplot(212)
        plt.plot(series_vals,[x["cos2phi_stddev"] for x in stackdata],"-bo")
        plt.ylabel('Std Dev Cos^2(PHI)')
        plt.xlabel(x_lab)

        plt.figure(2)
        plt.subplot(211)
        plt.plot(series_vals,[x["asp_rat_mean"] for x in stackdata],"-bo")
        plt.ylabel('Mean Aspect Ratio')
        plt.xlabel(x_lab)
        plt.subplot(212)
        plt.plot(series_vals,[x["asp_rat_stddev"] for x in stackdata],"-bo")
        plt.ylabel('Std Dev Aspect Ratio')
        plt.xlabel(x_lab)

        plt.figure(3)
        plt.plot(numreg,time_elapsed,"ro")
        plt.ylabel('Process Time (s)')
        plt.xlabel('Number of Features')
        plt.show()
        
# -------------------------------------------------------------

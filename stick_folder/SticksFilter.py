from math import ceil, pi, floor
from json.tool import main
import cv2
from cv2 import imwrite
import numpy as np
import copy

# ============================= EDGE DETECTOR CODE COPIED OVER ================================

# This method is responsible for generating a gaussian kernel, the size of the kernel is determined from sigma. Returns a 2d matrix
def get_gaus_kernel(sigma):
    # Formula to determine the size of the kernel matrix
    hsize = int(2 * np.ceil(3*sigma) + 1)

    # Generate a row ranging the entire matrix
    matrix_row = np.linspace(-(hsize//2),hsize//2,hsize)

    # Use the gaussian formula equation to calculate the guassian distribution on the row matrix
    gauss_row =  np.exp(-matrix_row**2/(2*(sigma**2)))/(sigma*np.sqrt(2*pi))

    # Use np.outer to multiply the two row matrixies into a 2d matrix 
    gauss_kern = np.outer(gauss_row,gauss_row)
    
    # Standardize the kernel ensuring that the values don't increase the brightness 
    gauss_kern /= np.sum(gauss_kern)
    
    # Return gaussian kernel 
    return gauss_kern

# Method is responsbile for doing a REFLECT type padding where it uses the outter rows and columns to pad the image.
# Returns the padded image
def pad_image(image,pad):
    padded_image = image
    for top in range(pad):
        # Use the top row to stack on the top of the image.
        padded_image = np.vstack((padded_image[0],padded_image))
        # Uses bottom row to stack on the bottom of the image
        padded_image = np.vstack((padded_image,padded_image[-1]))
    for rows in range(pad):
        # Use the left column to insert on the left.
        padded_image= np.insert(padded_image,0,padded_image[:,0],axis=1)
        # Use the right column to insert on the right.
        padded_image = np.insert(padded_image,padded_image.shape[1],padded_image[:,-1],axis=1)
    return padded_image

# This method is responsible for applying a convolution using any kernel.
def conv_filter(image, filter):
    # Create the blank array for the convolution output.
    conv_image = np.zeros(image.shape)

    #print(conv_image.shape)
    # Calculate the required amount of padding needed to keep the original dimensions of the image. 
    pad = filter.shape[0]//2

    # Pad the image using the pad image method. The image is padded by replicating the last row at the end of the image.
    # Doing so lets us optimize by not needing to check the pixel bounds every  
    padded_image = pad_image(image,pad)

    #cv2.imwrite("pad_view.jpg",padded_image)
    #print(padded_image.shape[1])
    #print(pad)
    
    # Iterate through the image using the kernal 

    # Iterate through the padded image starting and stopping in the areas of the orignal image.
    for x in range(pad,  image.shape[1]+pad):
        for y in range(pad,  image.shape[0]+pad):
            # Set conv value to zero before iterating through filter. 
            filter_value = 0
            # Iterate through the kernel, starting at negative to check the surrounding pixels at x and y.
            for x_f in range(-pad, pad+1):
                for y_f in range(-pad, pad+1):
                    # Convolution calculation for each pixel done by multiplying with the filter value.
                    # Add pad to the filter to account for padding
                    filter_value += padded_image[y_f+y][x_f+x]*filter[y_f+pad][x_f+pad]
            # Place the filter value into the convolution image which will be returned. 
            conv_image[y-pad][x-pad] = filter_value
    # Return the conv image
    return conv_image

# Method is responsbile for determining if a number falls between two numbers. Returns a boolean
def between(middle, bottom,top):
    if bottom <= middle < top:
        return True
    else:
        return False

# Method is responsbile for determining the edges depending on the angles obtained from the arctan of sobel edges.
# If the gradient falls within a range, it checks the designated pixels to determine if it sets it as an edge.  
# No threshold is applied in this method because the simple threshold at the end will handle removing lower value pixels. 
def non_max_filter(image,arctan):
    # Create the blank array for the non max suppression
    non_max = np.zeros(image.shape)

    # Convert all negative angles to positive 
    arctan[arctan < 0] += 180

    # Iterate through the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # Only check pixels which within the range of the image. 
            if(y+1 < image.shape[0] and y-1 >= 0 and x+1 < image.shape[1] and x-1 >= 0):
                # Obtain angle
                angle = arctan[y][x]
                # Checks if two pixels in the range are greater than or equal, to the middle pixel. 
                # If greater set value of non_max image to the same value from the edge in image  

                # Checking 90 Degree
                if between(angle,67.5,112.5) or between(angle,247.5,292.5):
                    if(image[y][x] >= image[y-1][x] and image[y][x] >= image[y+1][x]):
                        non_max[y][x] = image[y][x]
                # Checking 135 Degree
                elif between(angle,112.5,157.5) or between(angle,292.5,337.5):
                    if(image[y][x] >= image[y+1][x+1] and  image[y][x] >= image[y-1][x-1]):
                        non_max[y][x] = image[y][x]
                # check 0 degree
                elif between(angle,0,22.5) or between(angle,337.5,360) or between(angle,157.5,202.5):
                    if(image[y][x] >= image[y][x-1] and  image[y][x] >= image[y][x+1]):
                        non_max[y][x] = image[y][x]
                # Checking 45 Degree
                elif between(angle,22.5,67.5) or between(angle,202.5,247.5):
                    if(image[y][x] >= image[y+1][x-1] and  image[y][x] >= image[y-1][x+1]):
                        non_max[y][x] = image[y][x]
    # Return the non_max image, which now has the supressed edges.       
    return non_max

# This method is responsible for Simple Thresholding. 
def binary_threshold(image,threshold):
    # If the pixels are less than threshold then it is set to zero
    image[image < threshold] = 0
    # If the pixels are greater than threshold set them to 255 (max)
    image[image >= threshold] = 255
    # Return thresholded image. 
    return image

# ========================================== END =======================================================

#This method is responsible for adding a line between two points on a given kernel size.
def add_line(kernel,cord1,cord2):

    # Find the difference between two cordinates 
    dif_y = cord2[0] - cord1[0]  
    dif_x = cord2[1] - cord1[1]

    # Calculate the space between two points, selecting the larger difference required for drawing a line. 
    if abs(dif_x) > abs(dif_y):
        space_dif = abs(dif_x)
    else:
        space_dif = abs(dif_y)

    # Calculate the amount each value needs to increase in each space_dif loop
    x_inc = dif_x/float(space_dif)
    y_inc = dif_y/float(space_dif)
    
    # Place the first point before increasing the cur_x and cur_y values 
    cur_y = cord1[0]
    cur_x = cord1[1]
    kernel[cur_y][cur_x] = 1
    for val in range(space_dif):
        cur_x += x_inc
        cur_y += y_inc
        
        # Added this condition to replicate the filters in the pdf as the lines  
        if kernel.shape[0] > 5:
            matrix_y = round(cur_y)
        elif cur_y < kernel.shape[1]/2:
            matrix_y = floor(cur_y)
        else:
            matrix_y = ceil(cur_y)

        if kernel.shape[0] > 5:
            matrix_x = round(cur_x)
        elif cur_x < kernel.shape[1]/2:
            matrix_x = floor(cur_x)
        else:
            matrix_x = ceil(cur_x)
        kernel[matrix_y][matrix_x] = 1

    return kernel


# This method is used for selecting the corners to draw the line through the kernel
def generate_stick_kernel(kernel_size):
    # Stick filter only works with odd numbered kernels
    if kernel_size%2 != 1:
        kernel_size += 1
    kernel_list = []
    
    # cord 1 starts at the top image at the middle point
    cord_1 = [0,kernel_size//2]
    # cord 2 starts at the bottom of the image middle point
    cord_2 = [kernel_size-1,kernel_size//2]
    
    # Draw a line between the two points before going around the kernel, divide by 5 to maintain average between points on line.
    kernel_list.append(add_line(np.zeros((kernel_size,kernel_size)),cord_1,cord_2)/kernel_size)

    #Go left for cord 1 and right for cord 2
    for x in range((kernel_size//2)):
        cord_1[1] -= 1
        cord_2[1] += 1
        kernel_list.append(add_line(np.zeros((kernel_size,kernel_size)),cord_1,cord_2)/kernel_size)
    
    #Go down for cord 1 and up for cord 2
    for y in range((kernel_size-1)):
        cord_1[0] += 1
        cord_2[0] -= 1
        kernel_list.append(add_line(np.zeros((kernel_size,kernel_size)),cord_1,cord_2)/kernel_size)

    #Go right for cord 1 and left for cord 2 and stop before making the same line drawn at the start.
    for x in range((kernel_size//2)-1):
        cord_1[1] += 1
        cord_2[1] -= 1
        kernel_list.append(add_line(np.zeros((kernel_size,kernel_size)),cord_1,cord_2)/kernel_size)
    return kernel_list

# Stick filter is responsible for using a stick kernel to determine areas of high contrast.
# When finding the area of highest contrast determined by neighboring pixels it is multiplied with the angle of the image 
# and the new value is added to the image to increasing the contrast. 
def stick_filter(image, kernel_size):
    
    # Generate a list of stick kernels. 
    kernel_list = generate_stick_kernel(kernel_size)

    # Generate a kernel of 1's which will be used to calculate the average of all neighboring pixels. 
    neighboring_pixel_kern= np.ones((kernel_list[0].shape))
    #Divide by kernel_size^2
    neighboring_pixel_kern /= kernel_size**2


    #Create a blank edge list    
    edge_list = []
    
    # Calculate the output for all the edge kernels and add to edge_list
    for cur in kernel_list:
        new_image = cv2.filter2D(image,-1,cur)
        edge_list.append(new_image)
    
    # Calculate the values for all neighboring pixels.
    neighbor_conv = cv2.filter2D(image,-1,neighboring_pixel_kern)
    
    # Create a blank list for differences between edge_image and neighbor_conv
    difference_list = []

    # Calculate absolute difference between the edge and neighbor pixels
    for edge_image in edge_list:
        diff = abs(edge_image - neighbor_conv)
        difference_list.append(diff)
    
    # Select the values which provide the greatest difference to maximize contrast, place values in the max_dif image
    max_dif = np.zeros(image.shape)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for diff_img in difference_list:
                if diff_img[y,x] > max_dif[y,x]:
                    max_dif[y,x] = diff_img[y,x]

    cv2.imwrite("output/1_MaxDif_FilterLen_" + str(kernel_size) + ".png", max_dif)

    # Since it wasn't clear in the assignment pdf if we should stop at this point before doing non_max I outputted the image 
    # enhance the contrast of the sobel image using the diference between image and neighbors.

    # Find the difference between image nad neighbor then multiply by max dif to emphisize contrast
    difference_image = np.sign(image - neighbor_conv)*max_dif*2

    # Standardizing is applied to the images and added the difference, further enhancing the contrast.
    enhanced_image = (image/image.sum())*255 + difference_image
    
    return enhanced_image   
            
def main():
    # =============== USE THE SAME FILTERS FROM EdgeDetection.py =======================
    # Image must be in the same folder the script is being run or be given the exact file path. 
    image_file = "amogus.png"

    # The sigma value to determine the kernal size of the gaussian. 
    sigma_input = 0.6
    
    # Read in the image using the IMREAD_GRAYSCALE flag to make the image grayscale
    gray_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    #==========STEP 1===================
    # Gaussian blur
    gauss_kernel = get_gaus_kernel(sigma_input)
    blur_test = conv_filter(gray_image,gauss_kernel)

    #=========STEP 2====================
    # Use Sobel Filter for Edge Detection in both vertical and horizontal directions
    sobel_x_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #vertical
    sobel_y_kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) #horizontal
    sobel_x = conv_filter(blur_test,sobel_x_kernel)
    sobel_y = conv_filter(blur_test,sobel_y_kernel)
    combined_sobel = np.sqrt(sobel_x**2 +sobel_y**2)
    cv2.imwrite("output/0_Gradient_Magnitude.png", combined_sobel)
    # =========================== START OF StickFilters.py ===========================
    
    #For loop to test the usefulness of stick_filter and where it preforms best. n is the lengh of the stick. 
    n_list = [5,9,15]
    for n in n_list:
        #Call the stick_filter method 
        stick_map = stick_filter(combined_sobel, n)
        #Save the Enhanced sobel from the stick filter
        cv2.imwrite("output/2_Enhance_FilterLen_" + str(n) + ".png", stick_map)
        arctan_output = np.rad2deg(np.arctan2(sobel_y,sobel_x))
        non_max_output = non_max_filter(stick_map,arctan_output)
        #Apply non max to the stick_filter output.
        cv2.imwrite("output/3_NON_MAX_FilterLen_"+ str(n) +".png", non_max_output)
        #Apply threshold to the non max output. The threshold is lower as the image has less brightness but stronger edges and less noisy textures.
        thresh = binary_threshold(non_max_output,60)
        cv2.imwrite("output/4_threshold_FilterLen_"+ str(n) +".png", thresh)

if __name__ == "__main__":

    main()


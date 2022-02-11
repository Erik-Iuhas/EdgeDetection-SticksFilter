# Erik Iuhas 101076512 Part 2 Edge Detection. 

from cmath import pi
from json.tool import main
import cv2
import numpy as np

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
    
    # Normalize the kernel ensuring that the values don't increase the brightness 
    gauss_kern /= np.sum(gauss_kern)
    
    # Return gaussian kernel 
    return gauss_kern

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

# This method is responsible for Simple Thresholding. 
def binary_threshold(image,threshold):
    # If the pixels are less than threshold then it is set to zero
    image[image < threshold] = 0
    # If the pixels are greater than threshold set them to 255 (max)
    image[image >= threshold] = 255
    # Return thresholded image. 
    return image

def myEdgeFilter(gray_image,sigma):
    #==========STEP 1===================
    # Gaussian blur
    gauss_kernel = get_gaus_kernel(sigma)
    blur_test = conv_filter(gray_image,gauss_kernel)

    #=========STEP 2====================
    # Use Sobel Filter for Edge Detection in both vertical and horizontal directions
    sobel_x_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #vertical
    sobel_y_kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) #horizontal
    sobel_x = conv_filter(blur_test,sobel_x_kernel)
    sobel_y = conv_filter(blur_test,sobel_y_kernel)
    combined_sobel = np.sqrt(sobel_x**2 +sobel_y**2)
    
    #=========STEP 3====================
    # Using the sobel edge detection we generate a gradient angle image to do non-max suppression
    arctan_output = np.rad2deg(np.arctan2(sobel_y,sobel_x))
    non_max_output = non_max_filter(combined_sobel,arctan_output)
    cv2.imwrite("output/3_non_max_output.png",non_max_output)
    
    #=========STEP 4====================
    # Apply a simple threshold on non_max_output (I went with a Binary Threshold)
    edge_image = binary_threshold(non_max_output,100)

    #=========STEP 5====================
    # Save Images for analysis.
    cv2.imwrite("output/0_blur_output.jpg",blur_test)
    cv2.imwrite("output/1_sobel_gradient_mag.png",combined_sobel)
    cv2.imwrite("output/2_sobel_gradient_ori.png",arctan_output)
    
    
    # Used for testing comparison. 
    #cv_blur = cv2.GaussianBlur(gray_image,(7,7),1)
    #edge_test = cv2.Canny(gray_image,100,200)
    #cv2.imwrite("edge_baseline.jpg",edge_test)
    #cv2.imwrite("blur_baseline.jpg",cv_blur)

    
    #=========STEP 6====================
    #Return completed Edge Image
    return edge_image
    
def main():
    #Image must be in the same folder the script is being run or be given the exact file path. 
    image_file = "amogus.png"

    #The sigma value to determine the kernal size of the gaussian. 
    sigma_input = 0.6
    
    #Read in the image using the IMREAD_GRAYSCALE flag to make the image grayscale
    gray_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    #Filter Function which generates edge detected image. 
    edge_output = myEdgeFilter(gray_image,sigma_input)

    #Save edge detection image
    cv2.imwrite("output/4_Simple_Threshold.png",edge_output)


if __name__ == "__main__":
    main()


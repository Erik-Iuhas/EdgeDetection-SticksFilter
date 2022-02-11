Erik Iuhas 101076512
========================
EdgeDetection.py READ ME
========================

================= How to Run: =================================
To run the python file you want to ensure that you have OpenCV installed.

Open the terminal in the edge_folder directory and run the python file by typing "EdgeDetection.py" 
If there is an issue running you can also try "python EdgeDetection.py"

The script will run on the amogus.png as default

Ensure that when running the script that there is a folder named "output" in edge_folder, as this is where the images will be generated.
This is also where you'd be able to find existing examples of the edge detection on the amogus and cat2 example

How to run on a different image:
If you want to change the file you would need to go to the bottom of the script and change the image variable file name. 
================== Takeaways =================================
Purpose: 
The purpose of the edge detection is to detect edges in the image. In the code I start by using gaussian blur to 
remove any noise in the image. Afterwards a sobel edge detection kernel is applied to the entire image. Afterwards the 
gradient magnitude and orientation is obtained and used to generate non-maximum supression reducing down the edges in the 
image. Lastly a simple threshold is applied, in my case i used a binary threshold with an variable bound. 

======== Files contained: ============
- amogus.png (image)
- cat2.jpg (image)
- EdgeDetection.py (code)
- output (directory where images are saved)
- Edge detection (Directory with personally ran image results)
    -> Contains images for each step of the edge detection process.
    -> cat (directory) Included the cat output, gaussian is 0.6 and the threshold is using a value of 100.



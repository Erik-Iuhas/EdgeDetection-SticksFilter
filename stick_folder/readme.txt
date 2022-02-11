Erik Iuhas 101076512
========================
SticksFilter.py README
========================

========= How to Run:===============
To run the python file you want to ensure that you have OpenCV installed.

Open the terminal in the stick_folder directory and run the python file by typing "SticksFilter.py" 
If there is an issue running you can also try "python SticksFilter.py"

The script will run on the amogus.png as default

Ensure that when running the script that there is a folder named "output" in stick_folder, as this is where the images will be generated.
This is also where you'd be able to find existing examples of the edge detection on the amogus and cat2 example

How to run on a different image:
If you want to change the file you would need to go to the bottom of the script and change the image variable file. 

======================= Takeaways ==========================
Purpose: 
The purpose of the stick filter is to enhance the sobel_gradient's magnitude. I generate multiple images showing the steps 
the first step is the difference between the stick average and neighboring pixels, then using that to further add contrast to the image.

Assumption made: 
The line in the Assignment "The intensity should be increased along the direction of maximum response, amplifying local tonal differences" is very 
vauge in how it wants me to apply the solution. I referenced section 3.2[1] for increasing the intensity in sticks filter along the direction of maximum response,
personally when trying to apply non-maximum supression only to the difference of the neighbors and sticks average
the output wasn't useful and even worse than the original image inputted.

What I found:
As the sticks got longer (n = 15) the stronger edges increased more while also lowering noise
this overall increased the contrast and intensity of strong lines.

Citation:
[1] https://gigl.scs.carleton.ca/sites/default/files/david_mould/bnw-stick-2015.pdf

============ Files contained: ==============
- amogus.png (image)
- cat2.jpg (image)
- SticksFilter.py (code)
- output (directory)
- Edge detection (Directory with personally ran image results)
    -> Contains images for different outputs which had stick filters varying from n = 5,9,15.
    -> cat (directory) Included the cat output, the threshold is using a value of 100. Similar to part 2. 


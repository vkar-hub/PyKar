# PyKar
Simple karyotyping tool in python/opencv

Problem description:
  1. Resolving overlaps
      To separate overlapping chromosomes while retaining maximum resolution.
        Possible solutions:
        
          Solution A:
            Identify objects that are '+' or 'T' shaped
            Clone object
            Cut of corresponding arms
            (Maybe use houghlines?)
            
          Solution B:
            Allow user to select region
            Allow user copy and paste region
            Allow user to select and delete excess regions
            (Maybe be use grabcut OR build mini GUI in tk/qt)
            
          Solution C:
            Are other ways possible (?)
            
  2. Feature extraction
      Identify chromsomes based on other characteristics when sizes are roughly similar. 
        Identify banding pattern(?)
          How to recognize dark and light pixels of a foreground object?
          
        Identify centromere position (?)
          What makes centromere different from other regions of a chromosome in an image?
        
  3. Visualizing 
      Show all the chromosomes and allow user to rotate, switch and label/annotate according their analysis
        #Should learn how to do this

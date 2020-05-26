# Feature Matching tool
***
### Environment
    - Windows 10
    - Anaconda
       Recommend to use conda to create environment
       **conda env create -f environment.yml**
    
### Dependencies
    pip install Opencv
    pip install scipy

### Directory contents
    helpers.py
    memusecheck.py
    visualize.py
    student.py
    main.py

### RUNNING THE PROGRAM

in the directory open Anaconda prompt and type python main.py --p PAIR


### Running the program 
    The feature matchine system loads the image pair, extract features of each of them and match the corresponding ones

   
    1. get_interest_points
        - Use this function in the student class to get the interest points of the image using harris corner detector
        - Input: image, threshold , feature_width, k
        - Output: list of x,y coordinates of image interest points
   
    2. get_features 
        - Use student class get_features function for getting the feature descriptor of the interest point 
        - Input: image, list of interest points, feature_width
        - Output: 128 vcector length of the sift feature descriptor
        
    3. match_features
        - Use student class get_features function for getting the feature descriptor of the interest point 
        - Input: image 1 features, image 2 features
        - Output: matched features, confidence of matching
   
   4. Helper Functions
	euclidean_distance, get_neighbors, gradientMagnitudeandDirection, histogram
	
        
    * Note that, it is better to check the path for output file for each notebook to ensure that the notebook is working.


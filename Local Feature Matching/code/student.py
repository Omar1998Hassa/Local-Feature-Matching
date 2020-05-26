import numpy as np
import cv2
import math
from scipy.spatial import KDTree


def get_interest_points(image, feature_width, threshold, k, xIgnoreFromBegin=0,xIgnoreFromEnd=0,yIgnoreFromBegin=0,yIgnoreFromEnd=0):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    """
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    x = []
    y = []

    image_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    Ix, Iy = np.gradient(image_gaussian)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    row = image_gaussian.shape[0]
    column = image_gaussian.shape[1]

    offset = feature_width // 2

    for i in range(offset+yIgnoreFromBegin, row - offset-yIgnoreFromEnd):
        for j in range(offset+xIgnoreFromBegin, column - offset-xIgnoreFromEnd):
            Sxx = Ixx[i - offset:i + offset + 1, j - offset:j + offset + 1].sum()
            Syy = Iyy[i - offset:i + offset + 1, j - offset:j + offset + 1].sum()
            Sxy = Ixy[i - offset:i + offset + 1, j - offset:j + offset + 1].sum()

            det = (Sxx * Syy) - (Sxy * Sxy)
            trace = Sxx + Syy
            R = det - k * trace

            if R > threshold:
                x.append(i)
                y.append(j)

    # These are placeholders - replace with the coordinates of your interest points!
    x = np.array(x)
    y = np.array(y)
    return x, y


def get_features(image, x, y, feature_width):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    image_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    """
    Returns feature descriptors for a given set of interest points.


    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    features = []
    # print(len(x))
    for i in range(len(x)):
        thisKeyPointdescriptor = descriptor(x[i], y[i], image_gaussian)
        features.append(thisKeyPointdescriptor)
    # print(np.array(features).shape)
    # print(0)
    # print(np.array(features))
    # print(0)
    return np.array(features)


def gradientMagnitudeAndDirection(i, j, image):
    gradientMagnitude = (((image[i + 1, j] - image[i - 1, j]) ** 2 + (image[i, j + 1] - image[i, j - 1]) ** 2) ** 0.5)
    gradientOrientation = (180 / math.pi) * math.atan2((image[i, j + 1] - image[i, j - 1]),
                                                       (image[i + 1, j] - image[i - 1, j]))
    return gradientMagnitude, gradientOrientation


def histogram(i, j, image):
    i = int(i)
    j = int(j)
    hist = [0] * 8
    for b in range(i - 4, i):
        for c in range(j - 4, j):
            magnitude, theta = gradientMagnitudeAndDirection(b, c, image)
            while (theta < 0):
                theta = theta + 360
            if theta >= 0 and theta <= 45:
                hist[0] += magnitude
            if theta > 45 and theta <= 90:
                hist[1] += magnitude
            if theta > 90 and theta <= 135:
                hist[2] += magnitude
            if theta > 135 and theta <= 180:
                hist[3] += magnitude
            if theta > 180 and theta <= 225:
                hist[4] += magnitude
            if theta > 225 and theta <= 270:
                hist[5] += magnitude
            if theta > 270 and theta <= 315:
                hist[6] += magnitude
            if theta > 315 and theta <= 360:
                hist[7] += magnitude
    return hist


def descriptor(i, j, image):
    dis = [0] * 16
    dis[0] = histogram(i - 4, j - 4, image)
    dis[1] = histogram(i - 4, j, image)
    dis[2] = histogram(i - 4, j + 4, image)
    dis[3] = histogram(i - 4, j + 8, image)
    dis[4] = histogram(i, j - 4, image)
    dis[5] = histogram(i, j, image)
    dis[6] = histogram(i, j + 4, image)
    dis[7] = histogram(i, j + 8, image)
    dis[8] = histogram(i + 4, j - 4, image)
    dis[9] = histogram(i + 4, j, image)
    dis[10] = histogram(i + 4, j + 4, image)
    dis[11] = histogram(i + 4, j + 8, image)
    dis[12] = histogram(i + 8, j - 4, image)
    dis[13] = histogram(i + 8, j, image)
    dis[14] = histogram(i + 8, j + 4, image)
    dis[15] = histogram(i + 8, j + 8, image)
    return dis


# Locate the most similar neighbors
def get_neighbors(im1_feature, im2_features):
    minDistance1=1000
    minDistance1Index=-1
    minDistance2=1000
    minDistance2Index= -1
    for i in range(len(im2_features)):
        dist = euclidean_distance(im1_feature, im2_features[i, :, :])
        if(dist < minDistance1):
            minDistance2=minDistance1
            minDistance1=dist
            minDistance2Index=minDistance1Index
            minDistance1Index=i
    return  minDistance1,minDistance2,minDistance1Index


def euclidean_distance(im1_feature, im2_feature):
    sub = np.subtract(im1_feature, im2_feature)
    pow = np.multiply(sub, sub)
    the_euclidean_distance = np.sum(pow)
    return the_euclidean_distance


def match_features(im1_features, im2_features):
    """
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    # we are given lists of features like so: [{'Kp': ..., 'Descriptor': ...}, {'Kp':..., 'Descriptor': ...}, ...]
    # we extract the descriptors for both lists

    matches = []
    confidences = []

    tolerance = 0.4

    for i in range(len(im1_features)):
        minDistance1, minDistance2, minDistance1Index = get_neighbors(im1_features[i,:,:], im2_features)
        if (minDistance1/minDistance2)  < tolerance:
            matches.append((i,minDistance1Index))
            confidences.append(minDistance1/minDistance2)

    # return matches
    # These are placeholders - replace with your matches and confidences!
    return np.array(matches),np.array(confidences)

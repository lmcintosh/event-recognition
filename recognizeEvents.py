import numpy as np
import cv2
import brewer2mpl

from event-recognition-functions import *

movie_dir = '/Users/lmcintosh/Dropbox/Shared - Ian_Lane/Dataset'
movie_format = '.mov'
annotation_format = '.txt'
new_fps = 1.0
original_fps = 29.41176470588235
notableEvents = [str(i) for i in range(1,13) if i in [2,6,11,12]]

# get the full paths of all movies
moviePaths = getPaths(movie_dir, movie_format)
# get the full paths of all annotations
annotationPaths = getPaths(movie_dir, annotation_format)
# parse the annotation text files into dictionaries
annotationDicts = getAnnotations(movie_dir, annotationPaths, notableEvents, original_fps, new_fps)
# make a list of only the movies that have an annotation file
dataList = findMatching(movie_dir, movie_format, moviePaths, annotationDicts)


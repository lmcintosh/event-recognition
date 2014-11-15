import numpy as np
import cv2
import csv
from os import listdir
from scipy.signal import fftconvolve

# DATA HANDLING
def getPaths(directory, dotFileFormat):
    paths = []
    for f in listdir(directory):
        if f[-4:] == dotFileFormat:
            paths.append(directory + '/' + f)
    return paths

def loadFrames(filename, nframes=None, downsample=0):
    ''' Input:
    filename (full path or name of file in cwd)
    nframes (None or number of frames you want)
    downsample (0 to n, each time downsamples 2x)
    '''
    cap = cv2.VideoCapture(filename)
    data_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    if nframes is None:
        nframes = int(data_frames)
    else:
        nframes = int(np.minimum(nframes, cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    t = 0
    frames = None
    while(cap.isOpened() and t < nframes):
        ret, frame = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for _ in range(downsample):
            # Downsample by 2x per time
            gray = cv2.pyrDown(gray)
        if frames is None:
           frames = np.zeros( (nframes,) + gray.shape,dtype=np.uint8)
        frames[t] = gray
        t+=1
        if (t+1) % 1000 == 0:
            print '%d / %d' % (t+1, nframes)
    return frames.astype(np.float32), fps

def getAnnotations(annotationDir, annotationPaths, notableEvents, originalFPS, newFPS=1.0):
    ''' Returns a dictionary with Title and Frames'''
    annotations = []
    for a in annotationPaths:
        newMovieEvents = {}
        newMovieEvents['Frames'] = []
        title = a.replace(annotationDir + '/','')
        title = title.replace('.viratdata.events.txt','')
        newMovieEvents['Title'] = title
        with open(a) as inputfile:
            for row in csv.reader(inputfile):
                oneEvent = row[0].split()
                # check if event type matches our criterion
                if oneEvent[1] in ['2','6','11','12']:
                    # append start frame
                    currentEvents = newMovieEvents['Frames']
                    # convert to 1 fps frame number
                    newFrameNumber = int(round(newFPS*oneEvent[3]/originalFPS))
                    currentEvents.append(newFrameNumber)
                    newMovieEvents['Frames'] = currentEvents
        annotations.append(newMovieEvents)
    return annotations


def generateLabels(movieEventDict, nFrames):
    return [int(i in movieEventDict['Frames']) for i in xrange(nFrames)]


def findMatching(movieDir, movieFormat, moviePaths, annotations):
    ''' Returns a list of (moviePath, annotationDict) tuples'''
    data = []
    for a in annotations:
        associatedMoviePath = movieDir + '/' + a['Title'] + movieFormat
        try:
            associatedIndex = moviePaths.index(associatedMoviePath)
            data.append((moviePaths[associatedIndex],a))
        except:
            print associatedMoviePath + ' Not in path'
    return data



# FILTERING AND PREPROCESSING
# return the shape of a Gaussian
def gaussian(x=np.linspace(-5,5,50),sigma=1.,mu=0.):
     return np.array([(1./(2.*pi*sigma**2))*np.exp((-(xi-mu)**2.)/(2.*sigma**2)) for xi in x])

# return a 2d difference of Gaussians with zero mean and unit variance
def spatial_filter_2d(c_width=1.,s_width=2.,xs_num=50):
    xs = np.linspace(-5,5,xs_num)
    S = gaussian(x=xs, sigma=s_width)
    C = gaussian(x=xs, sigma=c_width)
    S = S/np.sum(S)
    C = C/np.sum(C)
    S_2d = zeros((len(S),len(S)))
    C_2d = zeros((len(C),len(C)))
    for idx,x in enumerate(xs):
        for idy,y in enumerate(xs):
            S_2d[idx,idy] = S[np.min([int(np.sqrt((idx-np.floor(len(xs)/2.))**2 
                                                     + (idy-np.floor(len(xs)/2.))**2) + np.floor(len(xs)/2.)), len(xs)-1])]
            C_2d[idx,idy] = C[np.min([int(np.sqrt((idx-np.floor(len(xs)/2.))**2 
                                                     + (idy-np.floor(len(xs)/2.))**2) + np.floor(len(xs)/2.)), len(xs)-1])]
    # Make all surrounds have same peak sensitivity of 1.
    S_2d = S_2d/np.max(S_2d)
    # Make center have same integral as surround
    C_2d = (np.sum(S_2d)/np.sum(C_2d))*C_2d
    X = C_2d - S_2d
    
    return X/np.sqrt(np.var(X))


def filterMovie(movie, filterBank):
    features = []
    for filt in filterBank:
        features.append([fftconvolve(im, filt, mode='same') for im in frames])

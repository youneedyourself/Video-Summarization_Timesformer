import faiss
from sklearn.metrics import pairwise_distances_argmin_min
import random
from utils import *

def offline(number_of_clusters, features):
    # Cluster the frames using K-Means

    # K-means from sklearn
    #kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(features)

    # K-means from faiss
    ncentroids = number_of_clusters
    niter = 10
    verbose = True
    x = features
    
    # Take the first dimension of the first element of the list
    dimension = x[0].shape[0]

    kmeans = faiss.Kmeans(dimension, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(x)

    #closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features)
    closest, _ = pairwise_distances_argmin_min(kmeans.centroids, x)
    
    closest_clips_frames = []

    for i in sorted(closest):
        for idx in range(i*8, (i+1)*8):
            closest_clips_frames.append(idx)

    return closest_clips_frames

def online(features, threshold, ratio):

    i = 0
    previous = i
    clips = []

    #compare the sum of squared difference between clips j and previous
    for j in range(1, len(features)):
        if sum_of_squared_difference(features[previous], features[j]) > threshold:
            clip = []

            # add frames from clip i to j-1 to the clip list
            for b in range(i*8, j*8):
                clip.append(b)

            # randomly select 15% of the frames from the clip list
            random_num = round(len(clip)*ratio/100)
            # sort the frames in the clip list to ensure the order of the frames
            random_Frames = sorted(random.sample(clip, random_num))
            i = j
            clips.extend(random_Frames)

        previous = j

    # add the last clip to the clip list
    clip = []
    if i==j:
        for c in range(j*8, j*8+8):
            clip.append(c)
            random_num = round(len(clip)*ratio/100)
            random_Frames = sorted(random.sample(clip, random_num))

    else: # (i<j)
        for c in range(i*8, (j+1)*8):
            clip.append(c)
            random_num = round(len(clip)*ratio/100)
            random_Frames = sorted(random.sample(clip, random_num))

    clips.extend(random_Frames)
        
    return clips

def test():
    pass
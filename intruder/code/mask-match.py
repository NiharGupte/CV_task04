import cv2
import numpy as np
import os
import argparse


def fast_detector(source, target, n=50):
    orb = cv2.ORB_create(n)
    source_keypoint = orb.detect(source, None)
    target_keypoint = orb.detect(target, None)
    return source_keypoint, target_keypoint


def dog_detector(source, target, n=50):
    sift = cv2.xfeatures2d.SIFT_create(n)
    source_keypoint = sift.detect(source, None)
    target_keypoint = sift.detect(target, None)
    return source_keypoint, target_keypoint


def sift_descriptor(source, source_keypoint, target, target_keypoint, n=50):
    sift = cv2.xfeatures2d.SIFT_create(n)
    source_descriptor = sift.compute(source, source_keypoint)[1]
    target_descriptor = sift.compute(target, target_keypoint)[1]
    return source_descriptor, target_descriptor


def brief_descriptor(source, source_keypoint, target, target_keypoint, n=50):
    orb = cv2.ORB_create(n)
    source_descriptor = orb.compute(source, source_keypoint)[1]
    target_descriptor = orb.compute(target, target_keypoint)[1]
    return source_descriptor, target_descriptor


def brute_force_distance(source_descriptor,target_descriptor):
    brute_force_obj = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    num_matches = brute_force_obj.match(source_descriptor, target_descriptor)
    num_matches = sorted(num_matches, key=lambda x: x.distance)[0:25]
    kp_distances = list(map(lambda x: x.distance, num_matches))
    # Taking the first 50 matches among the key points
    # for i in num_matches[:50]:
    #     kp_distances.append(i.distance)
    # # Average normalized distance
    # # average_distance = 0
    # average_distance = np.mean(kp_distances/np.max(kp_distances))
    average_distance = np.mean(kp_distances/np.max(kp_distances))
    return average_distance


def retrieval_scores(masked_image, detector, descriptor):
    function_call_dict = {

        "fast": fast_detector,
        "dog": dog_detector,
        "sift": sift_descriptor,
        "brief": brief_descriptor
    }
    database_files = os.listdir("../replicate/database_image/")
    image_kp_distances = {}
    for i in database_files:
        s = cv2.imread(os.path.join(r"..\replicate\database_image", i))
        t = masked_image
        kp1, kp2 = function_call_dict[detector](s, t, n=100)
        des1, des2 = function_call_dict[descriptor](s, kp1, t, kp2, n=100)
        distance = brute_force_distance(des1, des2)
        image_kp_distances[os.path.join("../replicate/database_image", i)] = 1 - distance# Adding the similairty index
    # return path_to_intruder, np.max(list(image_kp_distances.values()))
    return image_kp_distances


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    kp_list = ["fast", "dog"]
    des_list = ["sift", "brief"]
    trans_list = ["view", "scale", "rot", "light"]

    # function_call_dict = {
    #     "fast": fast_detector,
    #     "dog": dog_detector,
    #     "sift": sift_descriptor,
    #     "brief": brief_descriptor
    # }

    parser.add_argument("-i", "--i", help='''Should be path of "maskLow.jpg"''', required=True)
    parser.add_argument("-j", "--j", help='''Should be path of "maskMiddle?.jpg"''', required=True)

    args = parser.parse_args()

    masked_low_path = args.i
    masked_middle_path = args.j

    masked_low = cv2.imread(masked_low_path)

    masked_middle = cv2.imread(masked_middle_path)

    scores = retrieval_scores(masked_low, "dog", "sift")

    print("Reference similarity index : ", scores[os.path.join("../replicate/database_image", "middle.jpg")])

    reference = scores[os.path.join("../replicate/database_image", "middle.jpg")]

    scores_with_middle = retrieval_scores(masked_middle, "dog", "sift")

    potential_intruders = {}
    for i in scores_with_middle.keys():
        if reference < scores_with_middle[i]:
            potential_intruders[i] = scores_with_middle[i]

    print("Number of Potential Intruders : ", len(potential_intruders.keys()))

    if len(potential_intruders.keys()) > 0:
        index = np.argmax(potential_intruders.values())
        potential_intruders_list = list(potential_intruders.keys())
        print(potential_intruders_list)
        print("------------------------------------------------------------------------------------------------------")
        print("Most potential intruder : ", potential_intruders_list[index], "with similarity score : ", potential_intruders[potential_intruders_list[index]])
    else:
        print("No potential intruders")

















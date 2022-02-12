import cv2
import os
import argparse
import numpy as np


def surf_detector(source, target, n=50):
    surf = cv2.xfeatures2d.SURF_create(n)
    source_keypoint = surf.detect(source, None)
    target_keypoint = surf.detect(target, None)
    print(len(target_keypoint), n)
    return source_keypoint, target_keypoint

def boost_descriptor(source, source_keypoint, target, target_keypoint, n=50):
    boost = cv2.xfeatures2d.BoostDesc_create()
    source_descriptor = boost.compute(source, source_keypoint)[1]
    target_descriptor = boost.compute(target, target_keypoint)[1]
    return source_descriptor, target_descriptor


def brute_force(source, source_keypoint, source_descriptor, target, target_keypoint, target_descriptor):
    brute_force_obj = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    num_matches = brute_force_obj.match(source_descriptor, target_descriptor)
    num_matches = sorted(num_matches, key=lambda x: x.distance)
    # print("Descreiptor shape targ:", target_descriptor.shape,"Descreiptor shape source:", source_descriptor.shape, "Num_matches: ", len(num_matches))
    output_image = cv2.drawMatches(source, source_keypoint, target, target_keypoint, num_matches, None, flags=2)
    return output_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    kp_list = ["fast", "dog", "surf"]
    des_list = ["sift", "brief", "boost"]
    trans_list = ["view", "scale", "rot", "light"]

    function_call_dict = {
        "sift": sift_descriptor,
        "brief": brief_descriptor,
        "surf": surf_detector,
        "boost": boost_descriptor
    }

    parser.add_argument("-kp", "--kp", help='''Should be either of "fast" or "dog" ''', required=True)
    parser.add_argument("-des", "--des", help='''Should be either of "sift" or "brief" ''', required=True)
    parser.add_argument("-trans", "--trans", help='''Should be one of "view","scale","rot" or "light" ''',
                        required=True)
    parser.add_argument("-nm", "--nm", help='''Number of points to be matched ''', default=50, type=int)

    args = parser.parse_args()
    CLI_flag = (args.kp in kp_list) and (args.des in des_list) and (args.trans in trans_list) and (args.nm > 0)
    if not CLI_flag:
        print(CLI_flag)
        raise Exception("Invalid Command arguments : Command line arguments do not match the description. Enter again")

    source_img_name = args.trans + "_S.ppm"
    target_img_name = args.trans + "_T.ppm"
    N = args.nm

    S_orig = cv2.imread(os.path.join(r"..\data", source_img_name),0)
    T_orig = cv2.imread(os.path.join(r"..\data", target_img_name),0)

    S = np.zeros(S_orig.shape)
    T = np.zeros_like(T_orig)

    S = (S_orig-np.min(S_orig))/(np.max(S_orig) - np.min(S_orig))
    T = (T_orig-np.min(T_orig))/(np.max(T_orig) - np.min(T_orig))

    S = S_orig
    T = T_orig
    # S = cv2.imread("../data"+"/" +source_img_name)
    # T = cv2.imread("../data"+"/" +target_img_name)

    # print(source_img_name, cv2.imread("../data"+"/" +source_img_name), os.listdir(".."))

    kp1, kp2 = function_call_dict[args.kp](S, T, N)
    des1, des2 = function_call_dict[args.des](S, kp1, T, kp2, N)

    output = brute_force(S_orig, kp1, des1, T_orig, kp2, des2)

    cv2.imshow("Output (Press q to quit)", output)
    k = cv2.waitKey(0)

    while True:
        if k == ord('q'):
            break
    cv2.destroyAllWindows()

    output_name = args.kp + "_" + args.des + "_" + args.trans + "_" + str(args.nm) + ".png"
    output_path = os.path.join(r"..\results", output_name)
    cv2.imwrite(output_path, output)

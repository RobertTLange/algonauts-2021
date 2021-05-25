import os
import argparse
import zipfile
import numpy as np
from utils.helper import save_dict


def main():
    parser = argparse.ArgumentParser(description='Prepares submission for Algonauts 2021')
    parser.add_argument('-rd','--result_dir', help='contains predicted fMRI activity',
                        default = './results/alexnet_devkit/layer_5', type=str)
    parser.add_argument('-t','--track', help='mini_track - all ROIs, full_track - WB',
                        default = 'mini_track', type=str)
    args = vars(parser.parse_args())
    track = args['track']
    result_dir = args['result_dir']
    if track == 'full_track':
        ROIs = ['WB']
    else:
        ROIs = ['LOC','FFA','STS','EBA','PPA','V1','V2','V3','V4']

    num_subs = 10
    subs=[]
    for s in range(num_subs):
        subs.append('sub'+str(s+1).zfill(2))

    results = {}
    for ROI in ROIs:
        ROI_results = {}
        for sub in subs:
            ROI_result_file = os.path.join(result_dir,track,sub,ROI+"_test.npy")
            print("Result file path: ", ROI_result_file)
            if not os.path.exists(ROI_result_file):
                print("---------Warning : submission not ready ----------")
                print("Result not found for ",sub, " and ROI: ",ROI)
                return
            ROI_result = np.load(ROI_result_file)
            ROI_results[sub] = ROI_result
        results[ROI] = ROI_results

    save_dict(results,track+".pkl")
    zipped_results = zipfile.ZipFile(track+".zip", 'w')
    zipped_results.write(track+".pkl")
    zipped_results.close()


if __name__ == "__main__":
    main()

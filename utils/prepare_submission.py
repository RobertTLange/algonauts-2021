import os
import zipfile
import numpy as np
from helper import save_dict


def prepare_submission(model_name="alexnet_devkit", track = 'mini_track'):
    result_dir = f'./data/results/{model_name}/'
    if track == 'full_track': ROIs = ['WB']
    else: ROIs = ['LOC','FFA','STS','EBA','PPA','V1','V2','V3','V4']

    all_subjects = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05',
                    'sub06', 'sub07', 'sub08', 'sub09', 'sub10']

    # Load in individual test npy files (subject/roi)
    results = {}
    for ROI in ROIs:
        ROI_results = {}
        for sub in all_subjects:
            ROI_result_file = os.path.join(result_dir, track, sub,
                                           ROI + "_test.npy")
            print("Loaded result for {sub} - {ROI}: ", ROI_result_file)
            ROI_result = np.load(ROI_result_file)
            ROI_results[sub] = ROI_result
        results[ROI] = ROI_results

    # Create model_submission sub directory
    sub_dir = f"data/submissions/{model_name}/"
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # Save results as pkl and zip files
    save_dict(results, sub_dir +track + ".pkl")
    zipped_results = zipfile.ZipFile(sub_dir + track + ".zip", 'w')
    zipped_results.write(sub_dir + track + ".pkl")
    zipped_results.close()


if __name__ == "__main__":
    main()

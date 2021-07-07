import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mle_toolbox.visualize import visualize_2D_grid, plot_2D_heatmap


def plot_roi_subject_grid(hyper_log, title="MLP Encoder - Best AlexNet Layer",
                          min_heat=0.135, max_heat=0.35):
    heat_array, range_x, range_y = visualize_2D_grid(hyper_log,
                                  params_to_plot=["roi_type", "subject_id"],
                                  target_to_plot="best_bo_score",
                                  return_array=True)
    fig, ax = plot_2D_heatmap(range_x, range_y, heat_array,
                              title=title,
                              xy_labels=["Region of Interest", "Subject ID"],
                              variable_name="Correlation: fMRI - Encoder",
                              max_heat=max_heat, min_heat=min_heat)


def normalize_scores(heat_array):
    norm_matrix = np.array([[162, 69, 1034, 165, 120, 238, 249, 188, 60], #sub10
                            [191, 76, 1515, 262, 346, 271, 265, 245, 94],
                            [55, 163, 1244,	150, 306, 300, 238,	223, 85],
                            [101, 89, 1117, 33, 80, 195, 189, 174, 55],
                            [308, 119, 1356, 216, 173, 286, 281, 229, 108],
                            [309, 69, 1397, 210, 219, 326, 196, 176, 73],
                            [368, 210, 1153, 225, 398, 176, 209, 212, 117],
                            [376, 80, 1237, 368, 278, 164, 271, 270, 111],
                            [183, 157, 1348, 153, 421, 285, 231, 270, 95],
                            [351, 68, 1843, 425, 341, 232, 231, 261, 107]])
    heat2 = (heat_array * norm_matrix).sum(axis=0)/norm_matrix.sum(axis=0)
    return heat2

def get_norm_score(hyper_log, plot=False, min_heat=0.135, max_heat=0.36):
    heat_array, range_x, range_y = visualize_2D_grid(hyper_log,
                                  params_to_plot=["roi_type", "subject_id"],
                                  target_to_plot="best_bo_score",
                                  return_array=True)
    heat2 = normalize_scores(heat_array)
    print(heat2)
    if plot:
        fig, ax = plot_2D_heatmap(range_x, ["mean"], np.expand_dims(heat2, 0),
                                  title="#Voxel per Subject Weighted Score",
                                  xy_labels=["Region of Interest", "Score"],
                                  max_heat=max_heat, min_heat=min_heat)
    else:
        return heat2


def plot_average_scores(hyper_log):
    heat_array, range_x, range_y = visualize_2D_grid(hyper_log,
                                                 params_to_plot=["roi_type",
                                                                 "subject_id"],
                                                 target_to_plot="best_bo_score",
                                                 return_array=True)
    region_sub_mean = heat_array.mean(axis=0)
    fig, axs = plt.subplots(2, 1, figsize=(15,10))
    axs[0].bar(range_x, region_sub_mean)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_xlabel("Region of Interest")
    axs[0].set_ylabel("Correlation")
    axs[0].set_title("Subject-Meaned Correlation: fMRI - Encoder")

    region_roi_mean = heat_array[:, :-1].mean(axis=1)
    axs[1].bar(range_y, region_roi_mean)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_xlabel("Subject ID")
    axs[1].set_ylabel("Correlation")
    axs[1].set_title("ROI-Meaned Correlation: fMRI - Encoder")
    fig.tight_layout()

    # Sort scores for spreadsheet storage
    idx = []
    for v in ["V1", "V2", "V3", "V4", "LOC", "EBA", "FFA", "STS", "PPA"]:
        idx.append(np.where(range_x == v)[0][0])
    return region_sub_mean[idx]


def plot_bo_scores(meta_log, hyper_log, subject_id, roi_type,
                   num_bo_per_layer=50, num_layers=8):
    fig, ax = plt.subplots(figsize=(14,5))
    eval_id = hyper_log.hyper_log[hyper_log.hyper_log.subject_id == subject_id][hyper_log.hyper_log.roi_type == roi_type].run_id.iloc[0]
    ax.plot(np.arange(len(meta_log[eval_id].stats.best_bo_score.mean)),
            meta_log[eval_id].stats.best_bo_score.mean)
    for i in range(num_layers-1):
        ax.axvline(num_bo_per_layer + i*num_bo_per_layer,
                    ls="--", c="red", alpha=0.5)
        trans = ax.get_xaxis_transform()
        plt.text(num_bo_per_layer + i*num_bo_per_layer-num_bo_per_layer/2,
                 .5, 'Layer ' + str(i+1), fontsize=20,
                 transform=trans, rotation=90)
    plt.text(num_bo_per_layer + (num_layers-1)*num_bo_per_layer-num_bo_per_layer/2,
             .5, 'Layer ' + str(num_layers), fontsize=20,
             transform=trans, rotation=90)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("# Fitting Iteration")
    ax.set_ylabel("Correlation")
    #ax.set_ylim(0.1, 0.35)
    ax.set_title(f"Layerwise Bayesian Optimization: {subject_id} - {roi_type}")
    return


def plot_perf_per_layer(hyper_log, meta_log, num_layers=8,
                        num_bo_per_layer=20,
                        title="MLP Encoder - VGG Layer:",
                        min_heat=0.135, max_heat=0.35):
    df = {"subject_id": [], "roi_type": [],
          "layer_id": [], "best_bo_score": []}
    for e_t in range(len(hyper_log)):
        run_id = hyper_log.hyper_log.iloc[e_t].run_id
        subject_id = hyper_log.hyper_log.iloc[e_t].subject_id
        roi_type = hyper_log.hyper_log.iloc[e_t].roi_type
        results = meta_log[run_id].stats.best_bo_score.mean
        for i in range(num_layers):
            sub_layer_results = results[i*num_bo_per_layer: num_bo_per_layer + i*num_bo_per_layer]
            best_score_on_layer = np.max(sub_layer_results)
            df["subject_id"].append(subject_id)
            df["roi_type"].append(roi_type)
            df["layer_id"].append("layer_"+str(i+1))
            df["best_bo_score"].append(best_score_on_layer)

    layer_df = pd.DataFrame(df)
    #fig, axs = plt.subplots(2, 4, figsize=(50, 25))
    all_heats = []
    for i, l_id in enumerate(["layer_" + str(i+1) for i in range(num_layers)]):
        heat_array, range_x, range_y = visualize_2D_grid(layer_df,
                                      params_to_plot=["roi_type", "subject_id"],
                                      target_to_plot="best_bo_score",
                                      fixed_params={"layer_id": l_id},
                                      return_array=True)
        heat2 = normalize_scores(heat_array)
        all_heats.append(heat2)
    fig, ax = plot_2D_heatmap(range_x, ["L" + str(i+1) for i in range(num_layers)],
                              np.stack(all_heats, 0),
                              title=title,
                              xy_labels=["Region of Interest", "Layer"],
                              max_heat=max_heat, min_heat=min_heat)
    return


def plot_best_layer(hyper_log, meta_log, num_layers=5,
                    num_bo_per_layer=25,
                    title="Best VGG Layer per Subject/ROI:",
                    min_heat=0.1, max_heat=0.325):
    df = {"subject_id": [], "roi_type": [],
          "layer_id": [], "best_bo_score": []}
    for e_t in range(len(hyper_log)):
        run_id = hyper_log.hyper_log.iloc[e_t].run_id
        subject_id = hyper_log.hyper_log.iloc[e_t].subject_id
        roi_type = hyper_log.hyper_log.iloc[e_t].roi_type
        results = meta_log[run_id].stats.best_bo_score.mean
        for i in range(num_layers):
            sub_layer_results = results[i*num_bo_per_layer: num_bo_per_layer + i*num_bo_per_layer]
            best_score_on_layer = np.max(sub_layer_results)
            df["subject_id"].append(subject_id)
            df["roi_type"].append(roi_type)
            df["layer_id"].append("layer_" + str(i+1))
            df["best_bo_score"].append(best_score_on_layer)

    layer_df = pd.DataFrame(df)
    max_layer_idx = layer_df.groupby(['subject_id',
                                      'roi_type'])['best_bo_score'].transform(max) == df['best_bo_score']
    max_layer_df = layer_df[max_layer_idx]
    max_layer_df['best_layer_id'] = [int(l[-1]) for l in max_layer_df.layer_id]

    visualize_2D_grid(max_layer_df,
                      params_to_plot=["roi_type", "subject_id"],
                      target_to_plot="best_layer_id",
                      plot_title=title,
                      xy_labels=["Region of Interest", "Subject ID"],
                      variable_name="Layer ID")
    return


def plot_combined_scores(scores, range_x, range_y, title, ylabel,
                         min_heat=0.1, max_heat=0.325):
    fig, ax = plot_2D_heatmap(range_x, range_y,
                              np.stack(scores, 0),
                              title=title,
                              xy_labels=["Region of Interest", ylabel],
                              max_heat=max_heat, min_heat=min_heat)

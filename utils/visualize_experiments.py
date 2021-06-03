import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mle_toolbox.visualize import visualize_2D_grid


def plot_roi_subject_grid(hyper_log, title="MLP Encoder - Best AlexNet Layer"):
    visualize_2D_grid(hyper_log,
                      params_to_plot=["roi_type", "subject_id"],
                      target_to_plot="best_bo_score",
                      plot_title=title,
                      xy_labels=["Region of Interest", "Subject ID"],
                      variable_name="Correlation: fMRI - Encoder",
                      min_heat=0.1, max_heat=0.35)


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


def plot_bo_scores(meta_log, eval_id, subject_id, roi_type,
                   num_bo_per_layer=50, num_layers=8):
    fig, ax = plt.subplots(figsize=(10,5))
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
    ax.set_ylim(0.1, 0.35)
    ax.set_title(f"Layerwise Bayesian Optimization: {subject_id} - {roi_type}")
    return


def plot_perf_per_layer(hyper_log, meta_log, num_layers=8,
                        num_bo_per_layer=20,
                        title="MLP Encoder - VGG Layer:"):
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
    fig, axs = plt.subplots(2, 4, figsize=(50, 25))
    for i, l_id in enumerate(["layer_" + str(i+1) for i in range(num_layers)]):
        visualize_2D_grid(layer_df,
                          params_to_plot=["roi_type", "subject_id"],
                          target_to_plot="best_bo_score",
                          fixed_params={"layer_id": l_id},
                          plot_title=f"{title} {i+1}",
                          xy_labels=["Region of Interest", "Subject ID"],
                          variable_name="Correlation: fMRI - Encoder",
                          min_heat=0.04, max_heat=0.35,
                          fig=fig, ax=axs.flatten()[i])
    fig.tight_layout()
    return


def plot_best_layer(hyper_log, meta_log, num_layers=8,
                    num_bo_per_layer=50,
                    title="Best VGG Layer per Subject/ROI:"):
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

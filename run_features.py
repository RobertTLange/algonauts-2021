import argparse
from feature_extraction.generate_features import run_activation_features
from feature_compression.compress_features import run_compression


def generate_and_compress(model_type: str, filter_config: dict,
                          trafo_type: str, num_components: int):
    video_dir = 'data/AlgonautsVideos268_All_30fpsmax/'
    save_dir = (f'data/features/{model_type}/' +
                f'{filter_config["filter_name"]}/' +
                f'sr_{filter_config["sampling_rate"]}/')
    run_activation_features(model_type, save_dir, video_dir, filter_config)
    run_compression(save_dir, model_type, trafo_type, num_components)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_dir', '--experiment_dir',
                        default='experiments', help='Experiment Directory.')
    parser.add_argument('-model_type', '--model_type',
                        default='simclr_r50_2x_sk1_100pct',
                        help='Model to generate activations.')
    parser.add_argument('-sampling_rate', '--sampling_rate',
                        default=1, type=int, help='Video Sampling Rate.')
    parser.add_argument('-trafo_type', '--trafo_type',
                        default='pca', help='Compression Technique.')
    parser.add_argument('-filter_name', '--filter_name',
                        default='mean', help='Filter Technique.')
    parser.add_argument('-num_components', '--num_components',
                        default=50, type=int, help='Dimension to reduce to.')
    cmd_args = parser.parse_args()

    filter_config = {"filter_name": cmd_args.filter_name,
                     "sampling_rate": cmd_args.sampling_rate}
    generate_and_compress(cmd_args.model_type, filter_config,
                          cmd_args.trafo_type, cmd_args.num_components)

# python run_features.py --sampling_rate 1

# filter_configs = [
#                  {"filter_name": "raw",
#                  "sampling_rate": 4},
#                  {"filter_name": "mean",
#                   "sampling_rate": 4}
#                  {"filter_name": "1d-pca",
#                   "sampling_rate": 4},
#                  {"filter_name": "bold-kernel-1",
#                   "sampling_rate": 4,
#                   "peak": 20,
#                   "under": 40,
#                   'under_coeff': 0.35},
#                  {"filter_name": "bold-kernel-2",
#                   "sampling_rate": 4,
#                   "peak": 60,
#                   "under": 80,
#                   'under_coeff': 0.5},
#                  {"filter_name": "bold-kernel-3",
#                   "sampling_rate": 4,
#                   "peak": 40,
#                   "under": 60,
#                   'under_coeff': 0.35}
#                  ]

"""Main macro to draw rate plots for luminosity stability analysis."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import uproot
import matplotlib.pyplot as plt
import numpy as np
import yaml


def process_run(config: dict, run_name: str):  # pylint: disable=too-many-locals
    """
    Process a single run to draw trigger rates vs time.
    
    Parameters
    ----------
    config: dict
        Configuration dictionary.
    run_name: str
        Name of the run to process.
    """
    input_file = config["input"]
    detectors = config["detectors"]
    output_folder = config["output_dir"]
    bc_types = config["bc_types"]
    cross_sections = config["cross_sections"]

    with uproot.open(input_file) as file:
        lumi_folder = file["lumi-stability-p-p"]
        run_name = run_name.split(";1")[0]

        run_hists = {}
        run_ratios = {}

        # We store raw data in run_hists to make it picklable for return
        for detector in detectors:
            run_hists[detector] = {}
            run_ratios[detector] = {}
            for bc_type in bc_types:
                hist = lumi_folder[run_name][detector][f"{bc_type}/nBCsVsTime"]
                run_hists[detector][bc_type] = {
                    "values": hist.values(),
                    "edges": hist.axis().edges()
                }

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]})

        hist_data = {}
        for detector in detectors:
            for bc_type in bc_types:
                data = run_hists[detector][bc_type]
                values = data["values"]
                bin_edges = data["edges"]
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                hist_data[(detector, bc_type)] = (bin_centers, values)
                ax1.plot(
                    bin_centers,
                    values,
                    label=f"{detector} {bc_type}",
                    marker='o',
                    linestyle=""
                )

        ax1.set_xlim(0, 1440)
        ax1.set_xlabel(r'$t-t_\mathrm{SOF}$')
        ax1.set_ylabel(r'$\mathrm{N_{BC}}$')
        ax1.set_title(f'Run {run_name} - Bunch Crossings vs time')
        ax1.legend()
        ax1.grid(True, which="both", ls="--", lw=0.5)

        for bc_type in bc_types:
            bcs_vals = [hist_data[(d, bc_type)] for d in detectors]
            _, val_ref = bcs_vals.pop(0)
            for det, (bc, val) in zip(detectors[1:], bcs_vals):
                ratio_vals = val / (val_ref + 1e-10)
                ratio_vals *= (cross_sections[detectors[0]] / cross_sections[det])
                non_zero_indices = ratio_vals != 0
                run_ratios[det][bc_type] = ratio_vals[non_zero_indices]
                ax2.plot(
                    bc[non_zero_indices],
                    run_ratios[det][bc_type],
                    label=f'{det}/{detectors[0]}, {bc_type}',
                    marker='o',
                    linestyle=""
                )

        ax2.set_xlim(0, 1440)
        ax2.set_xlabel(r'$t-t_\mathrm{SOF}$')
        ax2.set_ylabel('Ratio')
        ax2.grid(True, which="both", ls="--", lw=0.5)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{output_folder}/runs/run_{run_name}_bunch_crossings_vs_time.png")
        plt.close()

        return run_name, run_hists, run_ratios

def draw_trigger_vs_time(config: dict, runs: list):
    """
    Draw trigger rates vs time for each run in parallel.
    
    Parameters
    ----------
    config: dict
        Configuration dictionary.
    runs: list
        List of run names.

    Returns
    -------
    hists: dict
        Dictionary containing histograms for each run.
    ratios: dict
        Dictionary containing ratio histograms for each run.
    """
    hists = {}
    ratios = {}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_run, config, run) for run in runs]
        for future in futures:
            run_name, run_hist, run_ratio = future.result()
            hists[run_name] = run_hist
            ratios[run_name] = run_ratio

    return hists, ratios


def draw_ratio_vs_run(ratios: dict, config: dict):  # pylint: disable=too-many-locals, too-many-statements
    """
    Draw the mean and standard deviation of the ratio vs cumulative run time.
    
    Parameters
    ----------
    ratios: dict
        Dictionary containing ratio histograms for each run.
    config: dict
        Configuration dictionary.
    """
    detectors = config["detectors"]
    bc_types = config["bc_types"]
    output_folder = config["output_dir"]

    for det in detectors[1:]:
        runs = {bc: [] for bc in bc_types}
        ratio_means = {bc: [] for bc in bc_types}
        ratio_stds = {bc: [] for bc in bc_types}
        run_durations = {bc: [] for bc in bc_types}
        for bc_type in bc_types:


            sorted_runs = sorted(ratios.keys(), key=int)

            for run in sorted_runs:
                run_ratios = ratios[run]
                if det in run_ratios:
                    runs[bc_type].append(int(run))
                    ratio_means[bc_type].append(run_ratios[det][bc_type].mean())
                    ratio_stds[bc_type].append(run_ratios[det][bc_type].std())

                    hist = run_ratios[det][bc_type]
                    non_zero_indices = hist != 0
                    if non_zero_indices.any():
                        duration = len(non_zero_indices)
                        duration -= np.argmax(non_zero_indices[::-1]) # Trim trailing zeros
                        duration -= np.argmax(non_zero_indices) # Trim leading zeros
                        run_durations[bc_type].append(duration)
                    else:
                        run_durations[bc_type].append(0)

            durations = np.array(run_durations[bc_type])
            cum_sum = np.cumsum(durations)
            x_centers = cum_sum - (durations / 2.0)
            x_errors = durations / 2.0

        # Ratio plot
        plt.figure()
        for bc_type in bc_types:
            plt.errorbar(
                x_centers,
                ratio_means[bc_type],
                yerr=ratio_stds[bc_type],
                xerr=x_errors,
                fmt='o',
                capsize=0,
                elinewidth=1,
                label=f'{det}/{detectors[0]}, {bc_type}'
            )
        for i, run_id in enumerate(runs[bc_type]):
            plt.annotate(str(run_id),
                        (x_centers[i], ratio_means[bc_type][i]),
                        textcoords="offset points",
                        xytext=(-10, 10),
                        ha='center',
                        fontsize=8,
                        rotation=90)

        plt.gcf().set_size_inches((sum(durations)*0.0005, 7))
        plt.xlabel('Cumulative Run Time (min)')
        plt.ylabel('Ratio')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        # plt.gca().set_ylim(0.5, 1.5)
        plt.grid(True, which="both", ls="--", lw=0.5)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_folder}/ratio_{det}_{detectors[0]}_vs_run.png")
        plt.savefig(f"{output_folder}/ratio_{det}_{detectors[0]}_vs_run.pdf")
        plt.close()

        # Ratio - mean plot
        plt.figure()
        for bc_type in bc_types:
            plt.errorbar(
                x_centers,
                ratio_means[bc_type] - np.mean(ratio_means[bc_type]),
                yerr=ratio_stds[bc_type],
                xerr=x_errors,
                fmt='o',
                capsize=0,
                elinewidth=1,
                label=f'{det}/{detectors[0]}, {bc_type}'
            )

        for i, run_id in enumerate(runs[bc_type]):
            plt.annotate(str(run_id),
                        (x_centers[i], ratio_means[bc_type][i] - np.mean(ratio_means[bc_type])),
                        textcoords="offset points",
                        xytext=(-10, 10),
                        ha='center',
                        fontsize=8,
                        rotation=90)

        plt.gcf().set_size_inches((sum(durations)*0.0005, 7))
        plt.xlabel('Cumulative Run Time (min)')
        plt.ylabel('Ratio - Mean')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        # plt.gca().set_ylim(-0.5, 0.5)
        plt.grid(True, which="both", ls="--", lw=0.5)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_folder}/ratio_{det}_{detectors[0]}_vs_run_meansubtracted.png")
        plt.savefig(f"{output_folder}/ratio_{det}_{detectors[0]}_vs_run_meansubtracted.pdf")
        plt.close()


def run_analysis(config_path: str):
    """
    Main function to run the stability analysis.
    
    Parameters
    ----------
    config_path: str
        Path to the configuration YAML file.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    input_file = config["input"]
    output_folder = config["output_dir"]

    # Create output directories if they don't exist
    os.makedirs(f"{output_folder}/runs", exist_ok=True)

    with uproot.open(input_file) as file:
        runs = file["lumi-stability-p-p"].keys(recursive=False)

    _, ratios = draw_trigger_vs_time(config, runs)
    draw_ratio_vs_run(ratios, config)

    # Create .gitignore to ignore all files in output folder
    gitignore_path = f"{output_folder}/.gitignore"
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write("*\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw rate plots for luminosity stability.")
    parser.add_argument("config", type=str, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    run_analysis(args.config)

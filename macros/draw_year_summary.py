import argparse
import os
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import uproot
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import yaml

def get_all_run_durations(ratios_year):
    """
    Creates a master dictionary of {run_id: duration} for all runs found in the year.
    """
    master_durations = {}

    # Iterate through every label and LBC to find all possible runs
    for label in ratios_year:
        for lbc in ratios_year[label]:
            for no_t0_for_leading in (True, False):
                run_data = ratios_year[label][lbc][no_t0_for_leading]
                for run in run_data:
                    if run in master_durations:
                        continue
                    
                    # Calculate duration from the first available detector/bc_type
                    run_ratios = run_data[run]
                    valid_detectors = [d for d in run_ratios.keys() if len(run_ratios[d].keys()) != 0]
                    if not valid_detectors:
                        continue
                    
                    det = valid_detectors[0]
                    bc_type = list(run_ratios[det].keys())[0]
                    hist = run_ratios[det][bc_type]
                    
                    non_zero = hist != 0
                    if non_zero.any():
                        # Calculate duration by trimming leading/trailing zeros
                        start = np.argmax(non_zero)
                        end = len(non_zero) - np.argmax(non_zero[::-1])
                        master_durations[run] = end - start
                    else:
                        master_durations[run] = 0
                    
    return master_durations


def get_ratio_mean_std(ratios):
    ratio_means = deepcopy(ratios)
    ratio_stds = deepcopy(ratios)

    durations = []

    for label in ratios:
        for lbc in ratios[label]:
            for no_t0_for_leading in (True, False):
                run_ratios = ratios[label][lbc][no_t0_for_leading]
                for run in run_ratios:
                    for det in run_ratios[run]:
                        for bc_type in run_ratios[run][det]:
                            ratio_means[label][lbc][no_t0_for_leading][run][det][bc_type] = ratios[label][lbc][no_t0_for_leading][run][det][bc_type].mean()
                            ratio_stds[label][lbc][no_t0_for_leading][run][det][bc_type] = ratios[label][lbc][no_t0_for_leading][run][det][bc_type].std()

    return ratio_means, ratio_stds

def draw_ratio_vs_require_t0_for_leading(ratios, output_folder, bc_type_sel, master_durations):
    ratios_mean, ratios_std = get_ratio_mean_std(ratios)
    
    # Sort runs globally to define the x-axis timeline
    all_runs_sorted = sorted(master_durations.keys(), key=int)
    durations_list = np.array([master_durations[r] for r in all_runs_sorted])
    cum_sum = np.cumsum(durations_list)
    x_centers_master = cum_sum - (durations_list / 2.0)
    x_errors_master = durations_list / 2.0
    
    # Create a mapping for quick lookup: run_id -> index in the sorted x-axis
    run_to_idx = {run: i for i, run in enumerate(all_runs_sorted)}

    for label in ratios_mean:
        if label == "vdm": continue
        plt.figure()
        
        for lbc in sorted(ratios_mean[label].keys()):
            for no_t0_for_leading in (True, False):
                ratio_data = ratios_mean[label][lbc][no_t0_for_leading]
                
                # Identify reference detector (assumes consistent naming)
                first_run = list(ratio_data.keys())[0]
                ref_detector = [d for d in ratio_data[first_run] if len(ratio_data[first_run][d].keys()) == 0][0]

                means = {}
                stds = {}
                plot_indices = [] # Indices of runs present in THIS specific lbc/label

                for run in sorted(ratio_data.keys(), key=int):
                    idx = run_to_idx[run]
                    run_ratios = ratio_data[run]
                    
                    for det in run_ratios:
                        if det == ref_detector: continue
                        key = f"{det}/{ref_detector}"
                        
                        if key not in means:
                            means[key] = {bc: [] for bc in run_ratios[det]}
                            stds[key] = {bc: [] for bc in run_ratios[det]}
                        
                        if bc_type_sel in run_ratios[det]:
                            means[key][bc_type_sel].append(run_ratios[det][bc_type_sel])
                            stds[key][bc_type_sel].append(ratios_std[label][lbc][no_t0_for_leading][run][det][bc_type_sel])
                    
                    plot_indices.append(idx)

                # Plotting with filtered x-axis data
                for det_key in means:
                    if bc_type_sel in means[det_key] and len(means[det_key][bc_type_sel]) > 0:
                        plt.errorbar(
                            x_centers_master[plot_indices],
                            means[det_key][bc_type_sel],
                            yerr=stds[det_key][bc_type_sel],
                            xerr=x_errors_master[plot_indices],
                            fmt='o', label=f'{label}, {det_key}, LBC={lbc}, no_t0_for_leading={no_t0_for_leading}'
                        )

        plt.gcf().set_size_inches((sum(durations_list)*0.0005 + 10, 7))
        plt.xlabel('Cumulative Run Time (min)')
        plt.ylabel('Ratio')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_folder / f"year_summary_{label}_{bc_type_sel}.pdf")
        plt.close()

def draw_ratio_vs_lbc(ratios, output_folder, bc_type_sel, no_t0_for_leading_sel, master_durations):
    ratios_mean, ratios_std = get_ratio_mean_std(ratios)
    
    # Sort runs globally to define the x-axis timeline
    all_runs_sorted = sorted(master_durations.keys(), key=int)
    durations_list = np.array([master_durations[r] for r in all_runs_sorted])
    cum_sum = np.cumsum(durations_list)
    x_centers_master = cum_sum - (durations_list / 2.0)
    x_errors_master = durations_list / 2.0
    
    # Create a mapping for quick lookup: run_id -> index in the sorted x-axis
    run_to_idx = {run: i for i, run in enumerate(all_runs_sorted)}

    for label in ratios_mean:
        if label == "vdm": continue
        plt.figure()
        
        for lbc in sorted(ratios_mean[label].keys()):
            ratio_data = ratios_mean[label][lbc][no_t0_for_leading_sel]
            
            # Identify reference detector (assumes consistent naming)
            first_run = list(ratio_data.keys())[0]
            ref_detector = [d for d in ratio_data[first_run] if len(ratio_data[first_run][d].keys()) == 0][0]

            means = {}
            stds = {}
            plot_indices = [] # Indices of runs present in THIS specific lbc/label

            for run in sorted(ratio_data.keys(), key=int):
                idx = run_to_idx[run]
                run_ratios = ratio_data[run]
                
                for det in run_ratios:
                    if det == ref_detector: continue
                    key = f"{det}/{ref_detector}"
                    
                    if key not in means:
                        means[key] = {bc: [] for bc in run_ratios[det]}
                        stds[key] = {bc: [] for bc in run_ratios[det]}
                    
                    if bc_type_sel in run_ratios[det]:
                        means[key][bc_type_sel].append(run_ratios[det][bc_type_sel])
                        stds[key][bc_type_sel].append(ratios_std[label][lbc][no_t0_for_leading_sel][run][det][bc_type_sel])
                
                plot_indices.append(idx)

            # Plotting with filtered x-axis data
            for det_key in means:
                if bc_type_sel in means[det_key] and len(means[det_key][bc_type_sel]) > 0:
                    plt.errorbar(
                        x_centers_master[plot_indices],
                        means[det_key][bc_type_sel],
                        yerr=stds[det_key][bc_type_sel],
                        xerr=x_errors_master[plot_indices],
                        fmt='o', label=f'{label}, {det_key}, LBC={lbc}, no_t0_for_leading={no_t0_for_leading_sel}'
                    )

        plt.gcf().set_size_inches((sum(durations_list)*0.0005 + 10, 7))
        plt.xlabel('Cumulative Run Time (min)')
        plt.ylabel('Ratio')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_folder / f"year_summary_{label}_{bc_type_sel}_no_t0_for_leading_{no_t0_for_leading_sel}.pdf")
        plt.close()

def draw_ratio_vs_bc_type(ratios, output_folder, lbc_sel, no_t0_for_leading_sel, master_durations):
    ratios_mean, ratios_std = get_ratio_mean_std(ratios)
    
    # Sort runs globally to define the x-axis timeline
    all_runs_sorted = sorted(master_durations.keys(), key=int)
    durations_list = np.array([master_durations[r] for r in all_runs_sorted])
    cum_sum = np.cumsum(durations_list)
    x_centers_master = cum_sum - (durations_list / 2.0)
    x_errors_master = durations_list / 2.0
    
    # Create a mapping for quick lookup: run_id -> index in the sorted x-axis
    run_to_idx = {run: i for i, run in enumerate(all_runs_sorted)}

    for label in ratios_mean:
        if label == "vdm": continue
        plt.figure()
        
        ratio_data = ratios_mean[label][lbc_sel][no_t0_for_leading_sel]
        # Identify reference detector (assumes consistent naming)
        first_run = list(ratio_data.keys())[0]
        ref_detector = [d for d in ratio_data[first_run] if len(ratio_data[first_run][d].keys()) == 0][0]

        means = {}
        stds = {}
        plot_indices = [] # Indices of runs present in THIS specific lbc/label/no_t0_for_leading

        for run in sorted(ratio_data.keys(), key=int):
            idx = run_to_idx[run]
            run_ratios = ratio_data[run]
            
            for det in run_ratios:
                if det == ref_detector: continue
                key = f"{det}/{ref_detector}"
                
                if key not in means:
                    means[key] = {bc: [] for bc in run_ratios[det]}
                    stds[key] = {bc: [] for bc in run_ratios[det]}
                
                for bc_type in run_ratios[det]:
                    means[key][bc_type].append(run_ratios[det][bc_type])
                    stds[key][bc_type].append(ratios_std[label][lbc_sel][no_t0_for_leading_sel][run][det][bc_type])
            
            plot_indices.append(idx)

        # Plotting with filtered x-axis data
        for det_key in means:
            for bc_type in means[det_key]:
                plt.errorbar(
                    x_centers_master[plot_indices],
                    means[det_key][bc_type],
                    yerr=stds[det_key][bc_type],
                    xerr=x_errors_master[plot_indices],
                    fmt='o', label=f'{label}, {det_key}, bc_type={bc_type}, lbc={lbc_sel}, no_t0_for_leading={no_t0_for_leading_sel}'
                )

        plt.gcf().set_size_inches((sum(durations_list)*0.0005 + 10, 7))
        plt.xlabel('Cumulative Run Time (min)')
        plt.ylabel('Ratio')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_folder / f"year_summary_{label}_lbc_{lbc_sel}_no_t0_for_leading_{no_t0_for_leading_sel}.pdf")
        plt.close()

def draw_ratio_vs_label(ratios, output_folder, bc_type_sel, lbc_sel, no_t0_for_leading_sel, master_durations):
    ratios_mean, ratios_std = get_ratio_mean_std(ratios)

    # Sort runs globally to define the x-axis timeline
    all_runs_sorted = sorted(master_durations.keys(), key=int)
    durations_list = np.array([master_durations[r] for r in all_runs_sorted])
    cum_sum = np.cumsum(durations_list)
    x_centers_master = cum_sum - (durations_list / 2.0)
    x_errors_master = durations_list / 2.0
    
    # Create a mapping for quick lookup: run_id -> index in the sorted x-axis
    run_to_idx = {run: i for i, run in enumerate(all_runs_sorted)}

    plt.figure()
    for label in ratios_mean:
        # if label == "vdm": continue
        
        ratio_data = ratios_mean[label][lbc_sel][no_t0_for_leading_sel]
        # Identify reference detector (assumes consistent naming)
        first_run = list(ratio_data.keys())[0]
        ref_detector = [d for d in ratio_data[first_run] if len(ratio_data[first_run][d].keys()) == 0][0]

        means = {}
        stds = {}
        plot_indices = [] # Indices of runs present in THIS specific lbc/label/no_t0_for_leading

        for run in sorted(ratio_data.keys(), key=int):
            idx = run_to_idx[run]
            run_ratios = ratio_data[run]
            
            for det in run_ratios:
                if det == ref_detector: continue
                key = f"{det}/{ref_detector}"
                
                if key not in means:
                    means[key] = {bc: [] for bc in run_ratios[det]}
                    stds[key] = {bc: [] for bc in run_ratios[det]}

                means[key][bc_type_sel].append(run_ratios[det][bc_type_sel])
                stds[key][bc_type_sel].append(ratios_std[label][lbc_sel][no_t0_for_leading_sel][run][det][bc_type_sel])
            
            plot_indices.append(idx)

        # Plotting with filtered x-axis data
        for det_key in means:
            plt.errorbar(
                x_centers_master[plot_indices],
                means[det_key][bc_type_sel],
                yerr=stds[det_key][bc_type_sel],
                xerr=x_errors_master[plot_indices],
                    fmt='o', label=f'{label}, {det_key}, lbc={lbc_sel}, no_t0_for_leading={no_t0_for_leading_sel}, bc_type={bc_type_sel}'
                )
            
        if label == "vdm":
            # Draw box across whole data taking period
            for det_key in means:
                plt.fill_between(
                    [0, sum(durations_list)], 
                    means[det_key][bc_type_sel][0]-stds[det_key][bc_type_sel][0], 
                    means[det_key][bc_type_sel][0]+stds[det_key][bc_type_sel][0], 
                    color='gray', 
                    alpha=0.2, 
                    edgecolor='none',
                    label='vdM Reference'
                )
    plt.gcf().set_size_inches((sum(durations_list)*0.0005 + 10, 7))
    plt.xlabel('Cumulative Run Time (min)')
    plt.ylabel('Ratio')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder / f"year_summary_lbc_{lbc_sel}_no_t0_for_leading_{no_t0_for_leading_sel}_{bc_type_sel}.pdf")
    plt.close()

def draw_year_summary(year, config_analysis, config_downloader):
    """
    Draw the mean and standard deviation of the ratio vs cumulative run time.
    
    Parameters
    ----------
    ratios: dict
        Dictionary containing ratio histograms for each run.
    config: dict
        Configuration dictionary.
    """
    detectors = config_analysis["detectors"]
    bc_types = config_analysis["bc_types"]
    output_folder = Path(config_analysis["output_dir"]) / "summaries" / str(year)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get inputs
    input_folder = Path(config_analysis["output_dir"])
    labels = []
    for key in config_downloader:
        try:
            cfg_year = int(key.split("_")[0])
            if cfg_year == year:
                labels.append(key.split("_", 1)[1])
        except ValueError:
            continue

    lbcs_all = set()
    ratios = {}
    for label in labels:
        ratios[label] = {}
        label_folder = input_folder / f"{year}_{label}"
        lbcs = sorted([int(d.name.split("_")[1]) for d in label_folder.iterdir() if (label_folder / d.name).is_dir()])
        for lbc in lbcs:
            ratios[label][lbc] = {}
            lbcs_all.add(lbc)
            for no_t0_for_leading in (True, False):
                with open(label_folder / f"lbc_{lbc}" / f"no_t0_{no_t0_for_leading}" / "runs" / "ratios.pkl", "rb") as f:
                    ratios[label][lbc][no_t0_for_leading] = pickle.load(f)

    master_durations = get_all_run_durations(ratios)
    for bc_type in bc_types:
        draw_ratio_vs_require_t0_for_leading(ratios, output_folder, bc_type_sel=bc_type, master_durations=master_durations)

    for bc_type in bc_types:
        for no_t0_for_leading in (True, False):
            draw_ratio_vs_lbc(ratios, output_folder, bc_type_sel=bc_type, no_t0_for_leading_sel=no_t0_for_leading, master_durations=master_durations)

    for lbc in lbcs_all:
        for no_t0_for_leading in (True, False):
            draw_ratio_vs_bc_type(ratios, output_folder, lbc_sel=lbc, no_t0_for_leading_sel=no_t0_for_leading, master_durations=master_durations)

    for bc_type in bc_types:
        for lbc in lbcs_all:
            for no_t0_for_leading in (True, False):
                draw_ratio_vs_label(ratios, output_folder, bc_type_sel=bc_type, lbc_sel=lbc, no_t0_for_leading_sel=no_t0_for_leading, master_durations=master_durations)

def run_analysis(downloader_cfg_path, analysis_cfg_path):
    """
    Main function to run the stability analysis.
    
    Parameters
    ----------
    downloader_cfg_path: str
        Path to the downloader configuration YAML file.
    analysis_cfg_path: str
        Path to the analysis configuration YAML file.
    """
    with open(analysis_cfg_path, 'r') as file:
        cfg_analysis = yaml.safe_load(file)
    with open(downloader_cfg_path, 'r') as file:
        cfg_downloader = yaml.safe_load(file)

    input_folder = cfg_analysis["output_dir"]  # Output folder from draw_rate.py
    output_folder = cfg_analysis["output_dir"]

    years = set()
    for key in cfg_downloader:
        try:
            years.add(int(key.split("_")[0]))
        except ValueError:
            continue

    for year in years:
        draw_year_summary(year, cfg_analysis, cfg_downloader)

    # Create .gitignore to ignore all files in output folder
    gitignore_path = f"{output_folder}/.gitignore"
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write("*\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis based on a YAML configuration.")
    parser.add_argument("config_downloader", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("config_analysis", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    run_analysis(args.config_downloader, args.config_analysis)
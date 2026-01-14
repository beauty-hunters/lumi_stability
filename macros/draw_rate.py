import argparse
import uproot
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def draw_trigger_vs_time(lumi_folder, runs, detectors, output_folder):
    """
    Draw trigger rates vs time for given runs and detectors.

    Parameters:
    lumi_folder: Uproot folder containing luminosity data.
    runs (list): List of run names.
    detectors (list): List of detector names.
    output_folder (str): Folder path to save the output plots.
    """
    hists = {} # Not needed outside the loop for now, but used for global storage
    ratios = {} # Not needed outside the loop for now, but used for global storage
    for run in tqdm.tqdm(runs, desc="Processing runs"):
        run = run.split(";1")[0]  # Clean run name
        hists[run] = {}
        ratios[run] = {}
        run_hists = hists[run]
        run_ratios = ratios[run]
        # Get the detector information (only for leading bunches here)
        for detector in detectors:
            run_hists[detector] = lumi_folder[run][detector]["BC_L/nBCsVsTime"]
        
        # Plotting
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]})
        
        hist_data = {}
        for detector in detectors:
            hist = run_hists[detector]
            bin_edges = hist.axis().edges()
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            values = hist.values()
            hist_data[detector] = (bin_centers, values)

            ax1.plot(bin_centers, values, label=detector, marker='o', linestyle="")
        
        ax1.set_xlim(0, 1440)
        ax1.set_xlabel(r'$t-t_\mathrm{SOF}$')
        ax1.set_ylabel(r'$\mathrm{N_{BC}}$')
        ax1.set_title(f'Run {run} - Bunch Crossings vs time')
        ax1.legend()
        ax1.grid(True, which="both", ls="--", lw=0.5)
        
        # Ratio panel
        bcs_vals = [hist_data[d] for d in detectors]
        _, val_ref = bcs_vals.pop(0)
        for det, (bc, val) in zip(detectors[1:], bcs_vals):
            ratio_vals = val / (val_ref + 1e-10)  # Avoid division by zero
            non_zero_indices = ratio_vals != 0
            run_ratios[det] = ratio_vals[non_zero_indices]
            ax2.plot(bc[non_zero_indices], run_ratios[det], label=f'{det}/{detectors[0]}', marker='o', linestyle="")

        ax2.set_xlim(0, 1440)
        ax2.set_xlabel(r'$t-t_\mathrm{SOF}$')
        ax2.set_ylabel('Ratio')
        ax2.grid(True, which="both", ls="--", lw=0.5)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_folder}/run_{run}_bunch_crossings_vs_time.png")
        plt.close()
    return hists, ratios


def draw_ratio_vs_run(ratios, detectors, output_folder):
    for det in detectors[1:]:
        runs = []
        ratio_means = []
        ratio_stds = []
        run_durations = []

        # Sort runs
        sorted_runs = sorted(ratios.keys(), key=int)
        
        for run in sorted_runs:
            run_ratios = ratios[run]
            if det in run_ratios:
                runs.append(int(run))
                ratio_means.append(run_ratios[det].mean())
                ratio_stds.append(run_ratios[det].std())

                hist = run_ratios[det]
                non_zero_indices = hist != 0
                if non_zero_indices.any():
                    duration = len(non_zero_indices) - np.argmax(non_zero_indices[::-1]) - np.argmax(non_zero_indices)
                    run_durations.append(duration)
                else:
                    run_durations.append(0)


        durations = np.array(run_durations)
        cum_sum = np.cumsum(durations)
        x_centers = cum_sum - (durations / 2.0)
        x_errors = durations / 2.0

        plt.figure(figsize=(sum(durations)*0.0004, 7))
        plt.errorbar(x_centers, ratio_means, yerr=ratio_stds, xerr=x_errors, 
                     fmt='o', capsize=0, elinewidth=1, label=f'{det}/{detectors[0]}')

        for i, run_id in enumerate(runs):
            plt.annotate(str(run_id), 
                         (x_centers[i], ratio_means[i]),
                         textcoords="offset points", 
                         xytext=(-10, 10),
                         ha='center', 
                         fontsize=8, 
                         rotation=90)

        plt.xlabel('Cumulative Run Time (min)')
        plt.ylabel('Ratio')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        plt.grid(True, which="both", ls="--", lw=0.5)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_folder}/ratio_{det}_vs_run.png")
        plt.savefig(f"{output_folder}/ratio_{det}_vs_run.pdf")
        plt.close()


def run(input_file: str, output_folder: str):
    """
    Draw rate plots for luminosity stability.

    Parameters:
    input_file (str): Path to the input ROOT file.
    output_folder (str): Folder path to save the output plots.
    """
    detectors = ["FT0VTx", "FDD"]
    with uproot.open(input_file) as file:
        lumi_folder = file["lumi-stability-p-p"]
        runs = file["lumi-stability-p-p"].keys(recursive=False)

        hists, ratios = draw_trigger_vs_time(lumi_folder, runs, detectors, output_folder)
        draw_ratio_vs_run(ratios, detectors, output_folder)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw rate plots for luminosity stability.")
    parser.add_argument("input", type=str, help="Path to the input ROOT file.")
    parser.add_argument("output", type=str, help="Folder path to save the output plots.")
    args = parser.parse_args()

    run(args.input, args.output)

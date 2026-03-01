"""Main macro to draw rate plots for luminosity stability analysis."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import uproot
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import yaml

def get_mu(hist_trg_vs_bcid, hist_inspected_vs_bcid):
    """
    Calculate mu vs BCID and mean mu.
    
    Parameters
    ----------
    hist_trg_vs_bcid: uproot.behaviors.TH1
        The trigger vs BCID histogram from which to calculate mean mu.
    hist_inspected_vs_bcid: uproot.behaviors.TH1
        The inspected vs BCID histogram from which to calculate mean mu.
    
    Returns
    -------
    dict
        A dictionary containing mean mu values vs BCID.
    """
    values_trg, x_edges = hist_trg_vs_bcid.to_numpy()
    values_inspected, _ = hist_inspected_vs_bcid.to_numpy()
    mask = values_trg > 0
    mu = np.zeros_like(values_trg)
    mu[mask] = - np.log(1 - values_trg[mask] / (values_inspected[mask]))
    mean_mu = - np.log(1 - np.sum(values_trg) / np.sum(values_inspected))
    return ({
        "values": mu,
        "edges": x_edges
    }, mean_mu)

def draw_comparison(
    data: dict,
    run_name: int,
    out_file: str,
    xlabel: str,
    ylabel: str,
    ratio: dict=None
):
    """
    Draw comparison plots for a given run.

    Parameters
    ----------
    data: dict
        Dictionary containing histogram data for each detector and BC type.
    run_name: int
        The name of the run being processed.
    out_file: str
        The output file path where plots are saved.
    xlabel: str
        Label for the x-axis.
    ylabel: str
        Label for the y-axis.
    ratio: dict, optional
        Dictionary containing ratio values for each detector and BC type. If not provided, simple ratios will be plotted.
    """

    detectors = list(data.keys())
    bc_types = list(data[detectors[0]].keys())

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]})

    hist_data = {}
    for detector in detectors:
        for bc_type in bc_types:
            data_single = data[detector][bc_type]
            values = data_single["values"]
            bin_edges = data_single["edges"]
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            hist_data[(detector, bc_type)] = (bin_centers, values)
            ax1.plot(
                bin_centers,
                values,
                label=f"{detector} {bc_type}",
                marker='o',
                markersize=4,
                alpha=0.7,
                linestyle=""
            )

    ax1.set_xlim(data[detector][bc_type]["edges"][0], data[detector][bc_type]["edges"][-1])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(f'Run {run_name}')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", lw=0.5)

    for bc_type in bc_types:
        bcs_vals = [hist_data[(d, bc_type)] for d in detectors]
        _, val_ref = bcs_vals.pop(0)
        for det, (bc, val) in zip(detectors[1:], bcs_vals):
            if ratio is not None:
                ratio_vals = ratio[det][bc_type]
            else:
                ratio_vals = val / (val_ref + 1e-10)
            non_zero_indices = ratio_vals != 0
            ratios = ratio_vals[non_zero_indices]
            ax2.plot(
                bc[non_zero_indices],
                ratios,
                label=f'{det}/{detectors[0]}, {bc_type}',
                marker='o',
                markersize=4,
                alpha=0.7,
                linestyle=""
            )

    ax2.set_xlim(data[detectors[0]][bc_types[0]]["edges"][0], data[detectors[0]][bc_types[0]]["edges"][-1])
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Ratio')
    ax2.grid(True, which="both", ls="--", lw=0.5)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def draw_sl_nsl_ratios(data: dict, detector: str, run_name: str, obs: str, output_folder: str, cross_sections: dict=None, pileup_corr: dict=None):
    """
    Draw the ratio of SL to NSL for a given detector and run.

    Parameters
    ----------
    data: dict
        Dictionary containing histogram data for each detector and BC type.
    detector: str
        The name of the detector for which to draw the SL/NSL ratio.
    run_name: str
        The name of the run being processed.
    output_folder: str
        The output folder where plots are saved.
    """

    detectors = list(data.keys())
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]})

    for det in detectors:
        for bc_type in [f"BC_SL_{detector}", f"BC_NSL_{detector}"]:
            data_single = data[det][bc_type]
            values = data_single["values"]
            bin_edges = data_single["edges"]
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax1.plot(
                bin_centers,
                values,
                label=f"{det} {bc_type}",
                marker='o',
                markersize=4,
                alpha=0.7,
                linestyle=""
            )

        data_nonzero = data[det][f"BC_NSL_{detector}"]["values"] != 0
        ratio = np.zeros_like(data[det][f"BC_SL_{detector}"]["values"])
        ratio[data_nonzero] = data[det][f"BC_SL_{detector}"]["values"][data_nonzero] / (data[det][f"BC_NSL_{detector}"]["values"][data_nonzero] + 1e-10)
        if cross_sections is not None:
            ratio *= cross_sections[detectors[0]] / cross_sections[det]
        if pileup_corr is not None:
            ratio *= pileup_corr[det][f"BC_SL_{detector}"]["values"] / pileup_corr[det][f"BC_NSL_{detector}"]["values"]
        non_zero_indices = ratio != 0
        ratios = ratio[non_zero_indices]
        # non_outlier_indices = np.abs(ratios - ratios.mean()) < 3 * ratios.std()
        # ratios = ratios[non_outlier_indices]
        ax2.plot(
            bin_centers[non_zero_indices],
            ratios,
            label=f'{det} SL/NSL_{detector}',
            marker='o',
            markersize=4,
            alpha=0.7,
            linestyle=""
        )

    ax1.set_xlim(0, 3500)
    ax1.set_xlabel("BC ID")
    if obs == "mu":
        ax1.set_ylabel(r"$\mu$")
    elif obs == "ratio":
        ax1.set_ylabel(r"$N_{BC}$")
    ax1.set_title(f'Run {run_name}')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", lw=0.5)

    # ax2.set_xlim(data[detectors[0]][bc_types[0]]["edges"][0], data[detectors[0]][bc_types[0]]["edges"][-1])
    ax2.set_xlim(0, 3500)
    if obs == "ratio":
        ax2.set_ylim(0, 50)
    ax2.set_xlabel("BC ID")
    ax2.set_ylabel('Ratio')
    ax2.grid(True, which="both", ls="--", lw=0.5)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{output_folder}/runs/run_{run_name}_{obs}_{det}_SL_NSL_{detector}_ratio.png")
    plt.close()

def draw_l_nl_ratios(data: dict, run_name: str, obs: str, output_folder: str, cross_sections: dict=None, pileup_corr: dict=None):
    """
    Draw the ratio of SL to NSL for a given detector and run.

    Parameters
    ----------
    data: dict
        Dictionary containing histogram data for each detector and BC type.
    detector: str
        The name of the detector for which to draw the SL/NSL ratio.
    run_name: str
        The name of the run being processed.
    output_folder: str
        The output folder where plots are saved.
    """

    detectors = list(data.keys())
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]})

    for det in detectors:
        for bc_type in [f"BC_L", f"BC_NL"]:
            data_single = data[det][bc_type]
            values = data_single["values"]
            bin_edges = data_single["edges"]
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax1.plot(
                bin_centers,
                values,
                label=f"{det} {bc_type}",
                marker='o',
                markersize=4,
                alpha=0.7,
                linestyle=""
            )

        data_nonzero = data[det][f"BC_NL"]["values"] != 0
        ratio = np.zeros_like(data[det][f"BC_L"]["values"])
        ratio[data_nonzero] = data[det][f"BC_L"]["values"][data_nonzero] / (data[det][f"BC_NL"]["values"][data_nonzero] + 1e-10)
        if cross_sections is not None:
            ratio *= cross_sections[detectors[0]] / cross_sections[det]
        if pileup_corr is not None:
            ratio *= pileup_corr[det][f"BC_L"]["values"] / pileup_corr[det][f"BC_NL"]["values"]
        non_zero_indices = ratio != 0
        ratios = ratio[non_zero_indices]
        # non_outlier_indices = np.abs(ratios - ratios.mean()) < 3 * ratios.std()
        # ratios = ratios[non_outlier_indices]
        ax2.plot(
            bin_centers[non_zero_indices],
            ratios,
            label=f'{det} L/NL',
            marker='o',
            markersize=4,
            alpha=0.7,
            linestyle=""
        )

    ax1.set_xlim(0, 3500)
    ax1.set_xlabel("BC ID")
    if obs == "mu":
        ax1.set_ylabel(r"$\mu$")
    elif obs == "ratio":
        ax1.set_ylabel(r"$N_{BC}$")
    ax1.set_title(f'Run {run_name}')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", lw=0.5)

    # ax2.set_xlim(data[detectors[0]][bc_types[0]]["edges"][0], data[detectors[0]][bc_types[0]]["edges"][-1])
    ax2.set_xlim(0, 3500)
    ax2.set_xlabel("BC ID")
    ax2.set_ylabel('Ratio')
    ax2.grid(True, which="both", ls="--", lw=0.5)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{output_folder}/runs/run_{run_name}_{obs}_{det}_L_NL_ratio.png")
    plt.close()


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
    year = Path(input_file).parent.parent.name.split("_")[0]

    with uproot.open(input_file) as file:
        lumi_folder = file[f"lumi-stability-p-p"]
        run_name = run_name.split(";1")[0]

        run_hists_vs_time = {}
        run_ratios_vs_time = {}
        run_hists_vs_bcid = {}
        run_ratios_vs_bcid = {}
        tot_counts = {}
        integrated_ratio = {}
        run_mu = {}
        run_mu_ratios = {}
        run_mu_vs_bcid = {}
        run_mu_ratios_vs_bcid = {}
        run_corr_pileup = {}
        run_corr_pileup_vs_bcid = {}

        hist_fs = lumi_folder[run_name]["FillingScheme"]
        hist_tfs = lumi_folder[run_name]["TFsPerMinute"]
        hist_int_rate = lumi_folder[run_name]["InteractionRate"]
        edges_int_rate = hist_int_rate.axis().edges()
        int_rate = sum(np.array(hist_int_rate.values()) * np.array((edges_int_rate[1:] + edges_int_rate[:-1]) / 2)) / sum(hist_int_rate.values())
        # We store raw data in run_hists_vs_time to make it picklable for return
        for detector in detectors:
            run_hists_vs_time[detector] = {}
            run_ratios_vs_time[detector] = {}
            run_hists_vs_bcid[detector] = {}
            run_ratios_vs_bcid[detector] = {}
            tot_counts[detector] = {}
            integrated_ratio[detector] = {}
            run_mu[detector] = {}
            run_mu_vs_bcid[detector] = {}
            run_mu_ratios_vs_bcid[detector] = {}
            run_corr_pileup[detector] = {}
            run_corr_pileup_vs_bcid[detector] = {}
            for bc_type in bc_types:
                hist_vs_time = lumi_folder[run_name][detector][bc_type]["nBCsVsTime"]
                hist_vs_bcid = lumi_folder[run_name][detector][bc_type]["nBCsVsBCID"]
                run_hists_vs_time[detector][bc_type] = {
                    "values": hist_vs_time.values(),
                    "edges": hist_vs_time.axis().edges()
                }
                run_hists_vs_bcid[detector][bc_type] = {
                    "values": hist_vs_bcid.values(),
                    "edges": hist_vs_bcid.axis().edges()
                }
                tot_counts[detector][bc_type] = hist_vs_time.values().sum()
                run_mu_vs_bcid[detector][bc_type], run_mu[detector][bc_type] = get_mu(
                    lumi_folder[run_name][detector][bc_type]["nBCsVsBCID"],
                    lumi_folder[run_name][detector][bc_type]["nBCsInspectedVsBCID"]
                )
                run_corr_pileup_vs_bcid[detector][bc_type] = {
                    "values": run_mu_vs_bcid[detector][bc_type]["values"] / (1 - np.exp(-run_mu_vs_bcid[detector][bc_type]["values"])),
                    "edges": run_mu_vs_bcid[detector][bc_type]["edges"]
                }                
                run_mu_vs_bcid[detector][bc_type]["values"] / (1 - np.exp(-run_mu_vs_bcid[detector][bc_type]["values"]))
                run_corr_pileup[detector][bc_type] = run_mu[detector][bc_type] / (1 - np.exp(-run_mu[detector][bc_type]))

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]})

        hists = {"trg_vs_time": run_hists_vs_time, "trg_vs_bcid": run_hists_vs_bcid, "mu_vs_bcid": run_mu_vs_bcid, "mu_corr_vs_bcid": run_corr_pileup_vs_bcid}
        ratios = {"trg_vs_time": run_ratios_vs_time, "trg_vs_bcid": run_ratios_vs_bcid, "mu_vs_bcid": run_mu_ratios_vs_bcid, "mu_corr_vs_bcid": run_mu_ratios_vs_bcid}

        for idetector, detector in enumerate(detectors):
            for bc_type in bc_types:
                integrated_ratio[detector][bc_type] = tot_counts[detector][bc_type] / tot_counts[detectors[0]][bc_type]
                integrated_ratio[detector][bc_type] *= (cross_sections[year][detectors[0]] / cross_sections[year][detector])
                integrated_ratio[detector][bc_type] *= run_corr_pileup[detector][bc_type] / run_corr_pileup[detectors[0]][bc_type]
                for hist_name, hists_dict in hists.items():
                    data = hists_dict[detector][bc_type]["values"]
                    ref_data = hists_dict[detectors[0]][bc_type]["values"]
                    if idetector != 0:
                        ratio_vals = data / (ref_data + 1e-10)
                        if hist_name == "trg_vs_time":
                            ratio_vals *= (cross_sections[year][detectors[0]] / cross_sections[year][detector])
                            ratio_vals *= run_corr_pileup[detector][bc_type] / run_corr_pileup[detectors[0]][bc_type]
                        elif hist_name == "trg_vs_bcid":
                            ratio_vals *= (cross_sections[year][detectors[0]] / cross_sections[year][detector])
                            ratio_vals *= run_corr_pileup_vs_bcid[detector][bc_type]["values"] / run_corr_pileup_vs_bcid[detectors[0]][bc_type]["values"]
                        ratios[hist_name][detector][bc_type] = ratio_vals

        if "BC_SL_FT0" in bc_types and "BC_NSL_FT0" in bc_types:
            draw_sl_nsl_ratios(run_hists_vs_bcid, "FT0", run_name, "ratio", output_folder, cross_sections[year], run_corr_pileup_vs_bcid)
            draw_sl_nsl_ratios(run_mu_vs_bcid, "FT0", run_name, "mu", output_folder)

        if "BC_SL_FDD" in bc_types and "BC_NSL_FDD" in bc_types:
            draw_sl_nsl_ratios(run_hists_vs_bcid, "FDD", run_name, "ratio", output_folder, cross_sections[year], run_corr_pileup_vs_bcid)
            draw_sl_nsl_ratios(run_mu_vs_bcid, "FDD", run_name, "mu", output_folder)

        if "BC_L" in bc_types and "BC_NL" in bc_types:
            draw_l_nl_ratios(run_hists_vs_bcid, run_name, "ratio", output_folder, cross_sections[year], run_corr_pileup_vs_bcid)
            draw_l_nl_ratios(run_mu_vs_bcid, run_name, "mu", output_folder)

        for hist_name, hists_dict in hists.items():
            data = hists_dict
            ratio = ratios[hist_name]
            ylabel = r'$\mathrm{N_{BC}}$' 
            if hist_name == "mu_vs_bcid":
                ylabel = r'$\mu$'
            elif hist_name == "mu_corr_vs_bcid":
                ylabel = r'$\mathrm{corr}_\mathrm{pile\_up}$'
            draw_comparison(
                data,
                run_name,
                f"{output_folder}/runs/run_{run_name}_{hist_name}.png",
                xlabel=r'$t-t_\mathrm{SOF}$' if hist_name == "trg_vs_time" else 'BC ID',
                ylabel=ylabel,
                ratio=ratio
            )

        output_data = {
            "run_name": run_name,
            "hists_vs_time": run_hists_vs_time,
            "ratios_vs_time": run_ratios_vs_time,
            "hists_vs_bcid": run_hists_vs_bcid,
            "ratios_vs_bcid": run_ratios_vs_bcid,
            "tot_counts": tot_counts,
            "integrated_ratio": integrated_ratio,
            "mu_vs_time": run_mu,
            "mu_vs_bcid": run_mu_vs_bcid,
            "corr_pileup_vs_time": run_corr_pileup,
            "corr_pileup_vs_bcid": run_corr_pileup_vs_bcid,
            "int_rate": int_rate
        }

        return output_data

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

    output = {
        "hists_vs_time": {},
        "hists_vs_bcid": {},
        "ratios_vs_time": {},
        "ratios_vs_bcid": {},
        "tot_counts": {},
        "integrated_ratio": {},
        "mu_vs_time": {},
        "mu_vs_bcid": {},
        "corr_pileup_vs_time": {},
        "corr_pileup_vs_bcid": {},
        "int_rate": {}
    }

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_run, config, run) for run in runs]
        for future in futures:
            data = future.result()
            run_name = data["run_name"]
            output["hists_vs_time"][run_name] = data["hists_vs_time"]
            output["hists_vs_bcid"][run_name] = data["hists_vs_bcid"]
            output["ratios_vs_time"][run_name] = data["ratios_vs_time"]
            output["ratios_vs_bcid"][run_name] = data["ratios_vs_bcid"]
            output["tot_counts"][run_name] = data["tot_counts"]
            output["integrated_ratio"][run_name] = data["integrated_ratio"]
            output["mu_vs_time"][run_name] = data["mu_vs_time"]
            output["mu_vs_bcid"][run_name] = data["mu_vs_bcid"]
            output["corr_pileup_vs_time"][run_name] = data["corr_pileup_vs_time"]
            output["corr_pileup_vs_bcid"][run_name] = data["corr_pileup_vs_bcid"]
            output["int_rate"][run_name] = data["int_rate"]   

    return output

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
        run_ratios = {bc: [] for bc in bc_types}
        ratio_stds = {bc: [] for bc in bc_types}
        run_durations = {bc: [] for bc in bc_types}
        for bc_type in bc_types:


            sorted_runs = sorted(ratios.keys(), key=int)

            for run in sorted_runs:
                if det in ratios[run]:
                    runs[bc_type].append(int(run))
                    run_ratios[bc_type].append(ratios[run][det][bc_type])



        # Ratio plot
        plt.figure()
        for bc_type in bc_types:
            syst = np.std(run_ratios[bc_type])**2
            syst += (np.mean(run_ratios[bc_type]) - 1)**2
            plt.errorbar(
                range(len(runs[bc_type])),
                run_ratios[bc_type],
                xerr=[0.5] * len(runs[bc_type]),
                fmt='o',
                capsize=0,
                elinewidth=1,
                label=fr'{det}/{detectors[0]}, {bc_type}, $\sqrt{{RMS^2 + \Delta^2}} = {np.sqrt(syst):.2f}$'
            )
        # for i, run_id in enumerate(runs[bc_type]):
        #     plt.annotate(str(run_id),
        #                 (i, run_ratios[bc_type][i]),
        #                 textcoords="offset points",
        #                 xytext=(-10, 10),
        #                 ha='center',
        #                 fontsize=8,
        #                 rotation=90)

        plt.gcf().set_size_inches((len(runs[bc_types[0]])*0.0005 + 10, 7))
        plt.xlabel('Run index', fontsize=12)
        plt.ylabel('Luminosity ratio', fontsize=12)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        # plt.gca().set_ylim(0.5, 1.5)
        plt.grid(True, which="both", ls="--", lw=0.5)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_folder}/ratio_{det}_{detectors[0]}_vs_run.png")
        plt.savefig(f"{output_folder}/ratio_{det}_{detectors[0]}_vs_run.pdf")
        plt.close()

def draw_mu_vs_run(mus: dict, config: dict):  # pylint: disable=too-many-locals, too-many-statements
    """
    Draw the mu values vs cumulative run time.
    
    Parameters
    ----------
    mus: dict
        Dictionary containing mu values for each run.
    config: dict
        Configuration dictionary.
    """
    detectors = config["detectors"]
    bc_types = config["bc_types"]
    output_folder = config["output_dir"]

    for det in detectors:
        runs = {bc: [] for bc in bc_types}
        mu_values = {bc: [] for bc in bc_types}
        for bc_type in bc_types:

            sorted_runs = sorted(mus.keys(), key=int)
            for run in sorted_runs:
                run_mus = mus[run]
                if det in run_mus:
                    runs[bc_type].append(int(run))
                    mu_values[bc_type].append(run_mus[det][bc_type])

            x_centers = list(range(len(runs[bc_type])))
            x_errors = [0.5] * len(runs[bc_type])

        # Ratio plot
        plt.figure()
        for bc_type in bc_types:
            plt.errorbar(
                x_centers,
                mu_values[bc_type],
                yerr=None,
                xerr=x_errors,
                fmt='o',
                capsize=0,
                elinewidth=1,
                label=f'{det}, {bc_type}'
            )
        # for i, run_id in enumerate(runs[bc_type]):
        #     plt.annotate(str(run_id),
        #                 (x_centers[i], mu_values[bc_type][i]),
        #                 textcoords="offset points",
        #                 xytext=(-10, 10),
        #                 ha='center',
        #                 fontsize=8,
        #                 rotation=90)

        plt.gcf().set_size_inches((len(runs[bc_type])*0.0005 + 10, 7))
        plt.ylim(0, 0.8)
        plt.xlabel('Run index', fontsize=12)
        plt.ylabel(r'$\mu$', fontsize=12)
        # plt.gca().set_ylim(0.5, 1.5)
        plt.grid(True, which="both", ls="--", lw=0.5)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_folder}/mu_{det}_vs_run.png")
        plt.savefig(f"{output_folder}/mu_{det}_vs_run.pdf")
        plt.close()

def draw_ratio_vs_int_rate(ratios: dict, int_rate: dict, config: dict):  # pylint: disable=too-many-locals, too-many-statements
    """
    Draw the ratio values vs integrated rate.
    
    Parameters
    ----------
    ratios: dict
        Dictionary containing ratio values for each run.
    int_rate: dict
        Dictionary containing integrated rate values for each run.
    config: dict
        Configuration dictionary.
    """
    detectors = config["detectors"]
    bc_types = config["bc_types"]
    output_folder = config["output_dir"]

    for det in detectors[1:]:
        runs = {bc: [] for bc in bc_types}
        run_ratios = {bc: [] for bc in bc_types}
        for bc_type in bc_types:
            sorted_runs = sorted(ratios.keys(), key=int)
            for run in sorted_runs:
                if det in ratios[run]:
                    runs[bc_type].append(int(run))
                    run_ratios[bc_type].append(ratios[run][det][bc_type])

        int_rates = [int_rate[run] for run in sorted_runs]

        # Ratio plot
        plt.figure()
        for bc_type in bc_types:
            plt.plot(
                int_rates,
                run_ratios[bc_type],
                label=f'{det}/{detectors[0]}, {bc_type}',
                linestyle="none",
                marker='o'
            )
        # for i, run_id in enumerate(runs[bc_type]):
        #     plt.annotate(str(run_id),
        #                 (int_rates[i], run_ratios[bc_type][i]),
        #                 textcoords="offset points",
        #                 xytext=(-10, 10),
        #                 ha='center',
        #                 fontsize=8,
        #                 rotation=90)

        plt.gcf().set_size_inches((10, 7))
        plt.xlabel('Interaction rate (kHz)', fontsize=12)
        plt.ylabel('Luminosity ratio', fontsize=12)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        # plt.gca().set_ylim(0.5, 1.5)
        plt.grid(True, which="both", ls="--", lw=0.5)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_folder}/ratio_vs_int_rate{det}_{detectors[0]}_vs_run.png")
        plt.savefig(f"{output_folder}/ratio_vs_int_rate{det}_{detectors[0]}_vs_run.pdf")
        plt.close()

def run_corr_pileup(corr_mus: dict, config: dict):  # pylint: disable=too-many-locals, too-many-statements
    """
    Draw the mu values vs cumulative run time.
    
    Parameters
    ----------
    corr_mus: dict
        Dictionary containing mu correction values for each run.
    config: dict
        Configuration dictionary.
    """
    detectors = config["detectors"]
    bc_types = config["bc_types"]
    output_folder = config["output_dir"]

    for det in detectors:
        runs = {bc: [] for bc in bc_types}
        corr_values = {bc: [] for bc in bc_types}
        for bc_type in bc_types:

            sorted_runs = sorted(corr_mus.keys(), key=int)
            for run in sorted_runs:
                run_corr = corr_mus[run]
                if det in run_corr:
                    runs[bc_type].append(int(run))
                    corr_values[bc_type].append(run_corr[det][bc_type])

            x_centers = list(range(len(runs[bc_type])))
            x_errors = [0.5] * len(runs[bc_type])

        # Ratio plot
        plt.figure()
        for bc_type in bc_types:
            plt.errorbar(
                x_centers,
                corr_values[bc_type],
                yerr=None,
                xerr=x_errors,
                fmt='o',
                capsize=0,
                elinewidth=1,
                label=f'{det}, {bc_type}'
            )
        # for i, run_id in enumerate(runs[bc_type]):
        #     plt.annotate(str(run_id),
        #                 (x_centers[i], corr_values[bc_type][i]),
        #                 textcoords="offset points",
        #                 xytext=(-10, 10),
        #                 ha='center',
        #                 fontsize=8,
        #                 rotation=90)

        plt.gcf().set_size_inches((len(runs[bc_type])*0.0005 + 10, 7))
        plt.ylim(0, 1.45)
        plt.xlabel('Run index', fontsize=12)
        plt.ylabel(rf'$\mathrm{{corr}}^{{{det}}}_{{\mathrm{{pile\_up}}}}$', fontsize=12)
        # plt.gca().set_ylim(0.5, 1.5)
        plt.grid(True, which="both", ls="--", lw=0.5)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_folder}/corr_mu_{det}_vs_run.png")
        plt.savefig(f"{output_folder}/corr_mu_{det}_vs_run.pdf")
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
        runs = file[f"lumi-stability-p-p"].keys(recursive=False)

    data = draw_trigger_vs_time(config, runs)
    with open(f"{output_folder}/runs/hists.pkl", "wb") as f:
        pickle.dump(data["hists_vs_bcid"], f)
    with open(f"{output_folder}/runs/ratios.pkl", "wb") as f:
        pickle.dump(data["ratios_vs_bcid"], f)
    with open(f"{output_folder}/runs/integrated_ratios.pkl", "wb") as f:
        pickle.dump(data["integrated_ratio"], f)
    with open(f"{output_folder}/runs/mu.pkl", "wb") as f:
        pickle.dump(data["mu_vs_bcid"], f)
    with open(f"{output_folder}/runs/corr_pileup.pkl", "wb") as f:
        pickle.dump(data["corr_pileup_vs_bcid"], f)
    with open(f"{output_folder}/runs/int_rate.pkl", "wb") as f:
        pickle.dump(data["int_rate"], f)

    draw_ratio_vs_run(data["integrated_ratio"], config)
    draw_ratio_vs_int_rate(data["integrated_ratio"], data["int_rate"], config)
    draw_mu_vs_run(data["mu_vs_time"], config)
    run_corr_pileup(data["corr_pileup_vs_time"], config)

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

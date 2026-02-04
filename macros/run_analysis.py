import argparse
import yaml
import os


def create_config(cfg_downloader, cfg_analysis, key, lbc, no_t0_for_leading, out_folder):
    """
    Create a new configuration dictionary for analysis based on downloader config.
    Parameters:
    -----------
    cfg_downloader : dict
        Configuration dictionary from the downloader.
    cfg_analysis : dict
        Configuration dictionary for the analysis.
    key : str
        The key corresponding to the current dataset.
    lbc : int
        The LBC value for the current dataset.
    no_t0_for_leading : bool
        The no_t0_for_leading flag for the current dataset.
    out_folder : str
        The output folder path for the analysis.    
    """
    cfg_new = cfg_analysis.copy()

    input_analysis = cfg_downloader['output_folder']
    cfg_new['input'] = os.path.join(input_analysis, key, f"lbc_{lbc}", f"no_t0_{no_t0_for_leading}", "AnalysisResults.root")
    cfg_new['output_dir'] = os.path.join(out_folder)

    return cfg_new


def run_analysis(downloader_cfg_path, analysis_cfg_path):
    with open(analysis_cfg_path, 'r') as file:
        cfg_analysis = yaml.safe_load(file)
    with open(downloader_cfg_path, 'r') as file:
        cfg_downloader = yaml.safe_load(file)

    output_folder = "outputs"

    # Create all output directories
    for key in cfg_downloader:
        if key == 'output_folder':
            continue
        for entry in cfg_downloader[key]:
            lbc = entry['lbc']
            no_t0_for_leading = entry['no_t0_for_leading']
            out_folder = os.path.join(output_folder, key, f"lbc_{lbc}", f"no_t0_{no_t0_for_leading}")
            os.makedirs(out_folder, exist_ok=True)

            cfg_new = create_config(cfg_downloader, cfg_analysis, key, lbc, no_t0_for_leading, out_folder)

            # Dump new config to a temporary file
            temp_cfg_path = f"temp_config_{key}_lbc{lbc}.yml"
            with open(temp_cfg_path, 'w') as temp_file:
                yaml.dump(cfg_new, temp_file)

            # Run the analysis
            os.system(f"python3 macros/draw_rate.py {temp_cfg_path}")
            os.remove(temp_cfg_path)

    # Add .gitignore and .gitkeep to output folder if not present (in case output folder is deleted)
    gitignore_path = os.path.join(output_folder, ".gitignore")
    gitkeep_path = os.path.join(output_folder, ".gitkeep")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w', encoding="utf-8") as f:
            f.write("*\n")
            f.write("!.gitkeep\n")
            f.write("!.gitignore\n")
    if not os.path.exists(gitkeep_path):
        with open(gitkeep_path, 'w', encoding="utf-8") as f:
            f.write("")

    # After all analyses are done, run the summary drawing script
    os.system(f"python3 macros/draw_year_summary.py {downloader_cfg_path} {analysis_cfg_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis based on a YAML configuration.")
    parser.add_argument("config_downloader", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("config_analysis", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    run_analysis(args.config_downloader, args.config_analysis)
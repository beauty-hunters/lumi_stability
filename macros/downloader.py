"""
Script to download files from grid based on a YAML configuration.

The script generates a bash script that uses the `alien_cp` command to copy files
from the grid to a specified local output folder. The bash script is then executed
in a JAliEn shell environment.
"""
import argparse
import os
import yaml

def create_bash_script(script_path, commands):
    """
    Create a bash script to be executed in jalien shell.
    
    Parameters:
    -----------
    script_path : str
        Path where the bash script will be created.
    commands : list of str
        List of commands to be included in the bash script.
    """
    with open(script_path, 'w', encoding="utf-8") as script_file:
        script_file.write("#!/bin/bash\n")
        for command in commands:
            script_file.write(f"{command}\n")
    os.chmod(script_path, 0o755)

def download_file(cfg_path):
    """
    Main function to download files based on the provided YAML configuration.
    
    Parameters:
    -----------
    cfg_path : str
        Path to the YAML configuration file.
    """
    with open(cfg_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)

    output_folder = config['output_folder']

    commands = []

    for key in config:
        if key == 'output_folder':
            continue
        for entry in config[key]:
            lbc = entry['lbc']
            out_folder = os.path.join(output_folder, key, f"lbc_{lbc}")
            os.makedirs(out_folder, exist_ok=True)

            commands.append(
                f"alien_cp {entry['dir']}/AnalysisResults.root \
                    file:{out_folder}/AnalysisResults.root"
            )


    script_path = "download_files.sh"
    create_bash_script(script_path, commands)
    os.system(f"alienv setenv JAliEn-ROOT/latest -c ./{script_path}")
    os.remove(script_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files based on a YAML configuration.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    download_file(args.config)

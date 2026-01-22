import argparse
import os
import sys
import yaml

def create_bash_script(script_path, commands):
    with open(script_path, 'w') as script_file:
        script_file.write("#!/bin/bash\n")
        for command in commands:
            script_file.write(f"{command}\n")
    os.chmod(script_path, 0o755)

def download_file(cfg_path):

    with open(cfg_path, 'r') as file:
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

            commands.append(f"alien_cp {entry['dir']}/AnalysisResults.root file:{out_folder}/AnalysisResults.root")


    script_path = "download_files.sh"
    create_bash_script(script_path, commands)
    os.system(f"alienv setenv JAliEn-ROOT/latest -c ./{script_path}")
    os.remove(script_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files based on a YAML configuration.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    download_file(args.config)
import os
import shutil
import argparse
import json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()
    work_dir = args.dir
    for folder in os.listdir(work_dir):
        if folder[:4] != "logs": continue
        config_path = os.path.join(work_dir, folder, "config.json")
        if not os.path.exists(config_path):
            continue
        config = json.loads(json.load(open(config_path)))        
        if 'motifs_folder' in config and 'group-contrib' in config['motifs_folder']:
            print(folder)
        else:
            continue
        no_plot = True
        for predictor_log in os.listdir(os.path.join(work_dir, folder)):
            if predictor_log[:9] == 'predictor':
                bad = True
                for f in os.listdir(os.path.join(work_dir, folder, predictor_log)):
                    if f.endswith('.png'):
                        bad = False
                if bad: 
                    shutil.rmtree(os.path.join(work_dir, folder, predictor_log))            
            if not predictor_log.endswith('.png'): continue
            no_plot = False
        if no_plot:
            shutil.rmtree(os.path.join(work_dir, folder))

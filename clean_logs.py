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
        try:
            config = json.loads(json.load(open(os.path.join(work_dir, folder, "config.json"))))
        except FileNotFoundError:
            continue
        # if 'motifs_folder' in config and 'hopv' in config['motifs_folder']:
        #     print(folder)
        # else:
        #     continue
        if 'motifs_folder' in config:
            if 'ptc' in config['motifs_folder']:
                continue  
        no_plot = True
        for predictor_log in os.listdir(os.path.join(work_dir, folder)):
            if predictor_log[:9] == 'predictor':
                bad = True
                for f in os.listdir(os.path.join(work_dir, folder, predictor_log)):
                    if f.endswith('.png'):
                        bad = False
                predictor_folder = os.path.join(work_dir, folder, predictor_log)
                if bad: 
                    shutil.rmtree(predictor_folder)            
                else:
                    best_perf = float("inf")
                    for f in os.listdir(predictor_folder):
                        if 'predictor_ckpt' == f[:14]:
                            perf = f.split('_')[-1]
                            if perf[-3:] != '.pt':
                                breakpoint()
                            if float(perf[:-3]) < best_perf:
                                best_perf = float(perf[:-3])
                    for f in os.listdir(predictor_folder):
                        if 'predictor_ckpt' == f[:14]:
                            perf = f.split('_')[-1]
                            assert perf[-3:] == '.pt'
                            if float(perf[:-3]) > best_perf:
                                os.remove(os.path.join(predictor_folder, f))
            if not predictor_log.endswith('.png'): continue
            no_plot = False
        if no_plot:
            shutil.rmtree(os.path.join(work_dir, folder))

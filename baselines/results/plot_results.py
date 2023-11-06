import json
import glob

datasets = ["Permeability", "group_ctb"]
methods = ["MolCLR_pretrained_gcn", "MolCLR_pretrained_gin"] #, "unimol"]

all_results = glob.glob("./*json")

for dtst in datasets:
    for mtd in methods:
        print("=============")
        res = None
        for rslt in all_results:
            if (dtst in rslt or dtst.lower() in rslt) and mtd in rslt:
                res = rslt
        
        try:
            assert res is not None
        except:
            import pdb; pdb.set_trace()
        
        res_dict = json.load(open(res, "r"))
        title = dtst + '-' + mtd
        
        print_str = f"{title:<40}" + f"{'mae' :<20}" + f"{'r^2' :<20}" + f"{'mse' :<20}" + "\n"
        
        for key in res_dict.keys():
            print_str += f"{key:<40}"
            mae_mean = res_dict[key]["mae_mean"]
            mae_std = res_dict[key]["mae_std"]
            r2_mean = res_dict[key]["r2_mean"]
            r2_std = res_dict[key]["r2_std"]
            mse_mean = res_dict[key]["rmse_mean"]
            mse_std = res_dict[key]["rmse_std"]
            
            print_str += f"{mae_mean: .4f}+/-{mae_std: .4f}".ljust(20)
            print_str += f"{r2_mean: .4f}+/-{r2_std: .4f}".ljust(20)
            print_str += f"{mse_mean: .4f}+/-{mse_std: .4f}".ljust(20)
            print_str += "\n"
        print(print_str)
        

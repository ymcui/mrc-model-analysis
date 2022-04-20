# coding=utf-8
import os
import csv
import json
import subprocess
import argparse

parser = argparse.ArgumentParser('Get results from prediction files.')
parser.add_argument('prediction_folder', metavar='prediction', help='prediction folder')
parser.add_argument('output_file', metavar='results.csv', help='output file')
parser.add_argument('--use-f1', '-f1', action='store_true')
OPT = parser.parse_args()

total_layer_num = 12
eval_metric = "f1" if OPT.use_f1 else "exact"

all_results = {"all":[], "q2":[], "q2p":[], "p2q":[], "p2":[]}
for mask_zone in ["all", "q2", "q2p", "p2q", "p2"]:
    for layer_idx in range(total_layer_num):
        target_file = os.path.join(OPT.prediction_folder, 'predictions_layer'+str(layer_idx)+"_"+mask_zone+".json")
        outline = subprocess.run('python evaluate-v2.0.py squad/dev-v1.1.json '+target_file, shell=True, stdout=subprocess.PIPE)
        outline_json = json.loads(outline.stdout.decode('utf-8'))
        all_results[mask_zone].append(str(round(outline_json[eval_metric], 3)))
    print("zone "+mask_zone+" is done!")

with open(OPT.output_file, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["layer", "all", "q2", "q2p", "p2q", "p2"])
    for k in range(total_layer_num):
        writer.writerow([k, all_results["all"][k], all_results["q2"][k], all_results["q2p"][k], all_results["p2q"][k], all_results["p2"][k]])
    csvfile.close()


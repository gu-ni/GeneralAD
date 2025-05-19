# %%
import os
import json
import numpy as np

# base

# GeneralAD
# HGAD

score_dir = "/workspace/MegaInspection/HGAD/scores"
scenarios = os.listdir(score_dir)
scenario_base_scores = {}

for scenario in scenarios:
    scenario_base_dir = os.path.join(score_dir, scenario, "base")
    if not os.path.isdir(scenario_base_dir):
        continue
    print(f"Processing {scenario} base")
    file = os.listdir(scenario_base_dir)[0]
    with open(os.path.join(scenario_base_dir, file), "r") as f:
        data = json.load(f)
    print(len(data))
    # Calculate final scores
    auroc_list = []
    ap_list = []
    for key, value in data.items():
        auroc_list.append(value["image_auroc"])
        ap_list.append(value["pixel_ap"])
    auroc_value = sum(auroc_list) / len(auroc_list)
    ap_value = sum(ap_list) / len(ap_list)
    auroc_value = auroc_value
    ap_value = ap_value
    
    scenario_base_scores[scenario] = {
        "image_auroc": auroc_value,
        "pixel_ap": ap_value,
    }
scenario_base_scores = dict(sorted(scenario_base_scores.items()))
scenario_base_scores_ = {
    s: {
        c: np.round(score * 100, 1) for c, score in case.items()
    } for s, case in scenario_base_scores.items()
}
scenario_base_scores_
# %%
# continual

import os
import json

scenarios = sorted(os.listdir(score_dir))

scenario_continual_scores = {}

for scenario in scenarios:
    cases = os.listdir(os.path.join(score_dir, scenario))
    cases = [case for case in cases if "base" not in case]
    cases = sorted(cases, key=lambda x: int(x[:-13]))

    fm_scores_of_dataset_per_case = {}

    for case in cases:
        case_dir = os.path.join(score_dir, scenario, case)
        datasets_path = [d for d in os.listdir(case_dir) if d.startswith("dataset")]
        datasets = sorted(datasets_path, key=lambda x: int(x[7:]))

        for dataset in datasets:
            model_scores = {}
            dataset_dir = os.path.join(case_dir, dataset)
            models = sorted(os.listdir(dataset_dir), key=lambda x: int(x.split(".")[0][5:]))
            model_json_list = [m for m in models if m.endswith("json")]
            
            if dataset == "dataset0":
                model_scores["model0"] = {
                    "image_auroc": scenario_base_scores[scenario]["image_auroc"],
                    "pixel_ap": scenario_base_scores[scenario]["pixel_ap"]
                }
                model_json_list.insert(0, "dataset0.json")

            for model in models:
                with open(os.path.join(dataset_dir, model), "r") as f:
                    data = json.load(f)
                
                auroc_list = [v["image_auroc"] for v in data.values()]
                ap_list = [v["pixel_ap"] for v in data.values()]

                avg_auroc = sum(auroc_list) / len(auroc_list)
                avg_ap = sum(ap_list) / len(ap_list)

                model_scores[model] = {
                    "image_auroc": avg_auroc,
                    "pixel_ap": avg_ap
                }
            
            # 모델명 순으로 정렬
            sorted_models = sorted(model_scores.keys(), key=lambda x: int(x.split(".")[0][5:]))

            # 중간 모델 최고 성능
            max_auroc = max(model_scores[m]["image_auroc"] for m in sorted_models)
            max_ap = max(model_scores[m]["pixel_ap"] for m in sorted_models)

            # 마지막 모델 성능
            last_model = sorted_models[-1]
            last_auroc = model_scores[last_model]["image_auroc"]
            last_ap = model_scores[last_model]["pixel_ap"]
            
            if len(model_json_list) > 1:
                # FM 계산
                fm_scores = {
                    "auroc_fm": max_auroc - last_auroc,
                    "ap_fm": max_ap - last_ap
                }
            else:
                fm_scores = None
                
            acc_scores = {
                "auroc_acc": last_auroc,
                "ap_acc": last_ap
            }

            if case not in fm_scores_of_dataset_per_case:
                fm_scores_of_dataset_per_case[case] = {}

            fm_scores_of_dataset_per_case[case][dataset] = (acc_scores, fm_scores)

    scenario_continual_scores[scenario] = fm_scores_of_dataset_per_case

# %%
avg_fm_scores = {}
for scenario, case_scores in scenario_continual_scores.items():
    avg_fm_scores[scenario] = {}
    for case, dataset_scores in case_scores.items():
        
        auroc_acc_list = []
        ap_acc_list = []
        auroc_fm_list = []
        ap_fm_list = []
        
        for key, value in dataset_scores.items():
            auroc_acc_list.append(value[0]["auroc_acc"])
            ap_acc_list.append(value[0]["ap_acc"])
            if value[1]:
                auroc_fm_list.append(value[1]["auroc_fm"])
                ap_fm_list.append(value[1]["ap_fm"])
            
        avg_auroc_acc = sum(auroc_acc_list) / len(auroc_acc_list)
        avg_ap_acc = sum(ap_acc_list) / len(ap_acc_list)
        avg_auroc_fm = sum(auroc_fm_list) / len(auroc_fm_list)
        avg_ap_fm = sum(ap_fm_list) / len(ap_fm_list)

        avg_fm_scores[scenario][case] = {
            "avg_auroc_acc": np.round(avg_auroc_acc * 100, 1),
            "avg_ap_acc": np.round(avg_ap_acc * 100, 1),
            "avg_auroc_fm": np.round(avg_auroc_fm * 100, 1),
            "avg_ap_fm": np.round(avg_ap_fm * 100, 1)
        }
avg_fm_scores
# %%
for scenario, value in avg_fm_scores.items():
    print(scenario)
    print(value)
    print()
# %%
avg_fm_scores["scenario_3"]
# %%



"""
GeneralAD

Scenario 1
85-5classes
48.6/1.9/25.3 -> 49.3/1.5/25.4 ||| 6.7/1.3/4.0 -> 5.5/1.2/3.4

85-10classes
50.4/1.6/26.0 -> 50.2/1.4/25.8 ||| 4.0/2.1/3.1 -> 3.2/1.5/2.4

85-30classes
46.6/1.4/24.0 -> 48.9/1.1/25.0 ||| 6.9/0.6/3.8 -> 5.8/1.2/3.5


Scenario 2
85-5classes
49.5/1.0/25.3 -> 49.0/0.8/24.9 ||| 8.6/2.6/5.6 -> 6.3/1.7/4.0

85-10classes
50.6/1.1/25.9 -> 51.3/0.9/26.1 ||| 4.2/1.7/3.0 -> 3.1/1.2/2.2

85-30classes
45.9/2.2/24.1 -> 47.7/2.1/24.9 ||| 7.0/0.0/3.5 -> 5.7/0.0/2.9

Scenario 3
85-5classes
53.4/1.0 -> 50.6/0.7 ||| 3.7/2.6 -> 3.3/2.0

85-10classes
52.2/1.5 -> 51.7/1.0 ||| 5.5/3.4 -> 5.3/2.6

85-30classes
51.4/1.9 -> 51.7/1.4 ||| 4.4/0.4 -> 3.3/0.9
"""




"""
GeneralAD
85-5classes (12tasks)
Scenario 1
48.6 -> 49.3 / 1.9 -> 1.5 ||| 6.7 -> 5.5 / 1.3 -> 1.2

Scenario 2
49.5 -> 49.0 / 1.0 -> 0.8 ||| 8.6 -> 6.3 / 2.6 -> 1.7

Scenario 3
53.4 -> 50.6 / 1.0 -> 0.7 ||| 3.7 -> 3.3 / 2.6 -> 2.0

85-10classes (6tasks)
Scenario 1
50.4/1.6/26.0 -> 50.2/1.4/25.8 ||| 4.0/2.1/3.1 -> 3.2/1.5/2.4

Scenario 2
50.6/1.1/25.9 -> 51.3/0.9/26.1 ||| 4.2/1.7/3.0 -> 3.1/1.2/2.2

Scenario 3
52.2/1.5 -> 51.7/1.0 ||| 5.5/3.4 -> 5.3/2.6

85-30classes (2tasks)
Scenario 1
46.6/1.4/24.0 -> 48.9/1.1/25.0 ||| 6.9/0.6/3.8 -> 5.8/1.2/3.5

Scenario 2
45.9/2.2/24.1 -> 47.7/2.1/24.9 ||| 7.0/0.0/3.5 -> 5.7/0.0/2.9

Scenario 3
51.4/1.9 -> 51.7/1.4 ||| 4.4/0.4 -> 3.3/0.9

"""



"""
HGAD
Scenario 1

85-5classes
53.9/6.1/30.0 -> 54.1/5.2/29.7 ||| 1.6/0.4/1.0 -> 1.5/0.4/1.0

85-10classes
53.2/5.4/29.3 -> 53.3/5.3/29.3 ||| 2.3/0.3/1.3 -> 2.1/0.3/1.2

85-30classes
52.0/5.7/28.9 -> 52.7/5.3/29.0 ||| 5.5/0.0/2.8 -> 4.8/0.0/2.4


Scenario 2

85-5classes
48.8/4.9/26.9 -> 51.1/4.3/27.7 ||| 2.3/0.2/1.3 -> 1.8/0.3/1.1

85-10classes
51.7/4.5/28.1 -> 51.8/4.5/28.2 ||| 1.6/0.2/0.9 -> 1.4/0.2/0.8

85-30classes
51.3/4.5/27.9 -> 51.8/4.3/28.1 ||| 3.0/0.1/1.6 -> 2.4/0.2/1.3


Scenario 3

85-5classes
52.8/4.2 -> 53.2/3.7 ||| 2.9/0.0 -> 2.5/0.1

85-10classes
52.9/4.0 -> 53.2/3.8 ||| 3.3/0.0 -> 2.9/0.0

85-30classes
51.7/4.3 -> 53.4/3.7 ||| 4.1/0.0 -> 2.7/0.0

"""
from pathlib import Path
import json
from random import random
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

import prepare_setup as ps
import ml_tracker.utilities.eva_embeddings as ev

import shared as shd
import occlusion as occl

runs = [
    (36,"3faf0d9266c1434ab553bcdb6aba07f0"),
    (37,"c4d4691ea56449b8bf004324f0d65b5f"),
    (38,"9e33a6b1e87642688bb3d20a927260f2"),
    (28,"2ff7de738e524c8493613c676cb9910a"),
    (28,"dc1787b74cc74ab8a75f65b06686349a"),
    (30,"c71ce1f121c94416a17afafa885059f9"),
    (30,"f4edc0c1909440bca950987d8de145fe"),
    (27,"5e43979bdbd94ba486dd963a5a55a507"),
    (29,"85b59ebd0db0465ab5a955b5c1c536bd"),
    (29,"b5f17c076c19411e98ecec6e49f53c08"),
    (38, "09ca9ad78fe343c2997c3ec47a77fdde"),
    (37, "4012e12f92a24e65bc2c5fb58a06492d"),
    (36, "954b1ba7dab64144972a55a8d05e39eb"),
    (40, "72704ed5b29f491fbc361b23de4b4ab4"),
    (41, "08eac58ec8c94e21afea824b86b8abb2"),
    (42, "604f99c7be8d470eba89fb9b79c9ec67"),
    (46, "ef6b164f6cb847579ddc85e9f1cec318"),
    (47, "723cd4bd5df944a1bb28fceada0077f0"),
    (48, "28135a00ac0c4933a417db943f734f8d"),
    (42, "0b54b5f516014a5c8d4c057e2aaa146a"),
    (41, "1a76387d25a846c09d559503e4578d5e"),
    (40, "c4f1afdaca844eb98702430ddff0904f"),
    (28, "ad535374b87b47b9ae6b26ac6bbba525"),
    (30, "254489b191824e6486fc1901adcfd9cf"),
    (29, "9aa08b5ad01645a5927f46f6f5633eb3"),
    (38, "b5b3c8a36de74b29b5420adeb44b7d22"),
    (37, "f03c609dd0384700b19d226ace451293"),
    (36, "56d64889a48d4548bba7e0a83b0c1ce3"),
    (47, "7213b8e73dca48dab8589d88472d17fb"),
    (48, "cd35a808f1b24657a83770d4fd77207a"),
    (46, "3e46c783f8e74a509099d1d41fde632e"),
    (42, "266a5e2e6a394ce298f5fe846c9ea346"),
    (41, "21686098fb8e49778e4f82b1a9b4f153"),
    (40, "c767c60054c54caebb9e2edeb515053b"),
    (47, "18c9b43215ae437c8801195b656db1b6"),
    (48, "e95b620b96094e3185eb94d650b3d379"),
    (46, "50930baae88f43b0ab9e490f5c698640")
]

#001
print("Start 001")
_, drifting_grating_dl_1 = shd.get_drifting_grating("single")
_, double_drifting_grating_dl_1 = shd.get_drifting_grating("double")
dls_1 = [
    ("drifting_grating_dl", drifting_grating_dl_1),
    ("double_drifting_grating_dl", double_drifting_grating_dl_1),
]

results = {}

for run in runs:
    path = Path("results") / "001" / f"{run[1]}.json"
    if path.exists():
        print("File already exists:", path)
        continue
    results[run[1]] = {}
    for dl in dls_1:
        results[run[1]][dl[0]] = {}
        for solver in ["logr"]:
            results[run[1]][dl[0]][solver] = {}
            for exp in range(1):
                mdl, opt, sched, train_dl, test_dl, epochs, test_per_epochs, auto_ckpt, cfg = ps.get_setup_from_config_v1_with_weights(run[0], run[1], file_name="self-supervised.safetensors", strict=True)
                blocks = len(cfg["model"]["blocks"].keys())
                all_res = {}
                z, y = shd.return_agg_encodings(mdl, dl[1], blocks=blocks)
                for k,v in z.items():
                    all_res[k] = shd.log_regr_multi_target_cv(v, y)
                results[run[1]][dl[0]][solver][exp] = all_res
    with open(f"results/001/{run[1]}.json", "w", encoding="utf-8") as f:
        json.dump(results[run[1]], f, indent=4)


#002
print("Start 002")
_, drifting_grating_dl_2 = shd.get_drifting_grating_masked("single")
_, double_drifting_grating_dl_2 = shd.get_drifting_grating_masked("double")
dls_2 = [
    ("drifting_grating_dl", drifting_grating_dl_2),
    ("double_drifting_grating_dl", double_drifting_grating_dl_2),
    ("test_dl", "test_dl")
]

results = {}

for run in runs:
    path = Path("results") / "002" / f"{run[1]}.json"
    if path.exists():
        print("File already exists:", path)
        continue
    results[run[1]] = {}
    for dl in dls_2:
        results[run[1]][dl[0]] = {}
        for solver in ["loss"]:
            results[run[1]][dl[0]][solver] = {}
            for exp in range(3):
                mdl, opt, sched, train_dl, test_dl, epochs, test_per_epochs, auto_ckpt, cfg = ps.get_setup_from_config_v1_with_weights(run[0], run[1], file_name="self-supervised.safetensors", strict=True)
                blocks = len(cfg["model"]["blocks"].keys())
                if isinstance(dl[1], str) and dl[1] == "test_dl":
                    dl = ("test_dl", test_dl)
                z = shd.return_loss(mdl, dl[1], blocks=blocks)
                results[run[1]][dl[0]][solver][exp] = z
    with open(f"results/002/{run[1]}.json", "w", encoding="utf-8") as f:
        json.dump(results[run[1]], f, indent=4)

#003
print("Start 003")
dls_3 = [
    ("test_dl", "test_dl")
]

results = {}

for run in runs:
    path = Path("results") / "003" / f"{run[1]}.json"
    if path.exists():
        print("File already exists:", path)
        continue
    results[run[1]] = {}
    for dl in dls_3:
        total_logr = {}
        for i in range(3):
            mdl, opt, sched, train_dl, test_dl, epochs, test_per_epochs, auto_ckpt, cfg = \
                ps.get_setup_from_config_v1_with_weights(
                    run[0],
                    run[1],
                    file_name="self-supervised.safetensors",
                    strict=True,
                )
            blocks = len(cfg["model"]["blocks"].keys())

            z_train, y_train = shd.return_agg_encodings(mdl, train_dl, blocks=blocks)
            z_test, y_test = shd.return_agg_encodings(mdl, test_dl, blocks=blocks)

            solvers = ["logr"]
            keys = list(z_train.keys())  # materialize keys once

            results_for_i = {solver: {} for solver in solvers}

            with ThreadPoolExecutor(max_workers=None) as executor:
                futures = [
                    executor.submit(shd.run_single_probe, solver, k, z_train, y_train, z_test, y_test)
                    for solver in solvers
                    for k in keys
                ]

                for future in as_completed(futures):
                    solver, k, res = future.result()
                    results_for_i[solver][k] = res

            total_logr[i] = deepcopy(results_for_i["logr"])

        results[run[1]][dl[0]] = {"logr": total_logr}
    with open(f"results/003/{run[1]}.json", "w", encoding="utf-8") as f:
        json.dump(results[run[1]], f, indent=4)


# 004
print("Start 004")
results = {}

for run in runs:
    path = Path("results") / "004" / f"{run[1]}.json"
    if path.exists():
        print("File already exists:", path)
        continue
    results[run[1]] = {}
    for dl in range(1):
        results[run[1]]["train_dl"] = {}
        results[run[1]]["test_dl"] = {}
        for solver in ["cos_sim"]:
            results[run[1]]["train_dl"][solver] = {}
            results[run[1]]["test_dl"][solver] = {}
            for exp in range(3):
                mdl, opt, sched, train_dl, test_dl, epochs, test_per_epochs, auto_ckpt, cfg = \
                    ps.get_setup_from_config_v1_with_weights(
                        run[0],
                        run[1],
                        file_name="self-supervised.safetensors",
                        strict=True,
                    )
                blocks = len(cfg["model"]["blocks"].keys())

                for dl in [("train_dl", train_dl), ("test_dl", test_dl)]:
                    z, _ = shd.return_encodings_sample(mdl, dl[1], blocks=blocks)
                    results[run[1]][dl[0]][solver][exp] = {k: float(shd.cos_patch_similarity(v, reduce="mean")) for k,v in z.items()}
                    del z
    with open(f"results/004/{run[1]}.json", "w", encoding="utf-8") as f:
        json.dump(results[run[1]], f, indent=4)

#005
print("Start 005")
_, drifting_grating_dl_5 = shd.get_drifting_grating_occl("single")
_, double_drifting_grating_dl_5 = shd.get_drifting_grating_occl("double")

dls_5 = [
    ("drifting_grating_dl", drifting_grating_dl_5),
    ("double_drifting_grating_dl", double_drifting_grating_dl_5),
]

results = {}

for run in runs:
    path = Path("results") / "005" / f"{run[1]}.json"
    if path.exists():
        print("File already exists:", path)
        continue
    results[run[1]] = {}
    for dl in dls_5:
        results[run[1]][dl[0]] = {}
        for solver in ["logr"]:
            results[run[1]][dl[0]][solver] = {}
            for exp in range(1):
                mdl, opt, sched, train_dl, test_dl, epochs, test_per_epochs, auto_ckpt, cfg = ps.get_setup_from_config_v1_with_weights(run[0], run[1], file_name="self-supervised.safetensors", strict=True)
                blocks = len(cfg["model"]["blocks"].keys())
                all_res = {}
                z, y = shd.return_agg_encodings(mdl, dl[1], blocks=blocks)
                for k,v in z.items():
                    all_res[k] = shd.log_regr_multi_target_cv(v, y)
                results[run[1]][dl[0]][solver][exp] = all_res
    with open(f"results/005/{run[1]}.json", "w", encoding="utf-8") as f:
        json.dump(results[run[1]], f, indent=4)

#006
print("Start 006")
dls_6 = [
    ("test_dl", "test_dl")
]

results = {}

for run in runs:
    path = Path("results") / "006" / f"{run[1]}.json"
    if path.exists():
        print("File already exists:", path)
        continue
    results[run[1]] = {}
    for dl in dls_6:
        total_logr = {}
        for i in range(3):
            mdl, opt, sched, train_dl, test_dl, epochs, test_per_epochs, auto_ckpt, cfg = \
                ps.get_setup_from_config_v1_with_weights(
                    run[0],
                    run[1],
                    file_name="self-supervised.safetensors",
                    strict=True,
                )
            blocks = len(cfg["model"]["blocks"].keys())

            z_train, y_train = shd.return_agg_encodings(mdl, train_dl, blocks=blocks, transform=occl.KeepOneRandomPatchOnlyUCF())
            z_test, y_test = shd.return_agg_encodings(mdl, test_dl, blocks=blocks, transform=occl.KeepOneRandomPatchOnlyUCF())


            solvers = ["logr"]
            keys = list(z_train.keys())  # materialize keys once

            results_for_i = {solver: {} for solver in solvers}

            with ThreadPoolExecutor(max_workers=None) as executor:
                futures = [
                    executor.submit(shd.run_single_probe, solver, k, z_train, y_train, z_test, y_test)
                    for solver in solvers
                    for k in keys
                ]

                for future in as_completed(futures):
                    solver, k, res = future.result()
                    results_for_i[solver][k] = res

            total_logr[i] = deepcopy(results_for_i["logr"])

        results[run[1]][dl[0]] = {"logr": total_logr}
    with open(f"results/006/{run[1]}.json", "w", encoding="utf-8") as f:
        json.dump(results[run[1]], f, indent=4)

#007
print("Start 007")
dls_7 = [
    # ("drifting_grating_dl", drifting_grating_dl),
    # ("double_drifting_grating_dl", double_drifting_grating_dl),
    ("test_dl", "test_dl")
]

results = {}

for run in runs:
    path = Path("results") / "007" / f"{run[1]}.json"
    if path.exists():
        print("File already exists:", path)
        continue
    results[run[1]] = {}
    for dl in dls_7:
        total_emb = {}
        if dl[0] != "test_dl":
            mdl, opt, sched, train_dl, test_dl, epochs, test_per_epochs, auto_ckpt, cfg = \
                ps.get_setup_from_config_v1_with_weights(
                    run[0],
                    run[1],
                    file_name="self-supervised.safetensors",
                    strict=True,
                )
            blocks = len(cfg["model"]["blocks"].keys())
            z, y = shd.return_agg_encodings(mdl, dl[1], blocks=blocks)


        for i in range(3):
            if dl[0] == "test_dl":
                mdl, opt, sched, train_dl, test_dl, epochs, test_per_epochs, auto_ckpt, cfg = \
                    ps.get_setup_from_config_v1_with_weights(
                        run[0],
                        run[1],
                        file_name="self-supervised.safetensors",
                        strict=True,
                    )
                blocks = len(cfg["model"]["blocks"].keys())
                z, y = shd.return_agg_encodings(mdl, test_dl, blocks=blocks)

            keys = list(z.keys())  # materialize keys once

            results_for_i = {"emb": {}}

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(shd.build_metrics, k, z, y)
                    for k in keys
                ]

                for future in as_completed(futures):
                    k, res = future.result()
                    results_for_i["emb"][k] = res

            total_emb[i] = deepcopy(results_for_i["emb"])

        results[run[1]][dl[0]] = {"emb": total_emb}
    with open(f"results/007/{run[1]}.json", "w", encoding="utf-8") as f:
        json.dump(results[run[1]], f, indent=4)

# 008
print("Start 008")
results = {}

for run in runs:
    path = Path("results") / "008" / f"{run[1]}.json"
    if path.exists():
        print("File already exists:", path)
        continue
    results[run[1]] = {}
    for dl in range(1):
        results[run[1]]["test_dl"] = {}
        for solver in ["linear_cka"]:
            results[run[1]]["test_dl"][solver] = {}
            for exp in range(3):
                mdl, opt, sched, train_dl, test_dl, epochs, test_per_epochs, auto_ckpt, cfg = \
                    ps.get_setup_from_config_v1_with_weights(
                        run[0],
                        run[1],
                        file_name="self-supervised.safetensors",
                        strict=True,
                    )

                blocks = len(cfg["model"]["blocks"].keys())

                for dl in [("test_dl", test_dl)]:
                    z, _ = shd.return_agg_encodings(mdl, dl[1], blocks)
                    results[run[1]][dl[0]][solver][exp] = {}
                    for i in range(len(z.keys())-1):
                        results[run[1]][dl[0]][solver][exp][i] = {}
                        results[run[1]][dl[0]][solver][exp][i].update(ev.linear_cka(z[i], z[i+1]))
                        results[run[1]][dl[0]][solver][exp][i].update(ev.rbf_cka(z[i], z[i+1]))
                    del z
    with open(f"results/008/{run[1]}.json", "w", encoding="utf-8") as f:
        json.dump(results[run[1]], f, indent=4)

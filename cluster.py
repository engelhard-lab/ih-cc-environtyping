import os
import argparse
import torch
import numpy as np
import pandas as pd
import pickle
import subprocess
from utils import yaml_config_hook
from modules import resnet34, network_stick, network_stick_nops, transform
from modules.NpyDataset_stick import NpyDataset


def inference(loader, model, device, ps):
    model.eval()
    feature_vector = []
    labels_vector = [] if ps else None  # Only store labels if `ps=True`

    for step, (x_i, _) in enumerate(loader):
        x = x_i.to(device)
        with torch.no_grad():
            c = model.forward_cluster(x)  # Cluster features
            c = c.detach()
            feature_vector.extend(c.cpu().numpy())

            if ps:
                g = model.forward_participant(x)  # Participant classification
                g = g.detach()
                labels_vector.extend(g.cpu().numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector) if ps else None

    print("Features shape:", feature_vector.shape)
    if ps:
        print("Labels shape:", labels_vector.shape)

    return (feature_vector, labels_vector) if ps else feature_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to the configuration file")
    args, unknown = parser.parse_known_args()

    config = yaml_config_hook.yaml_config_hook(args.config)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.demo:
        n_participant = len([name for name in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, name))])
    else:
        if not args.folders:
            args.folders = [int(name[13:-1]) for name in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, name))]
            n_participant = len(args.folders)
        else:
            n_participant = len(args.folders)

    print(f"Number of individuals in dataset: {n_participant}")
    print(f"Number of classes: {args.class_num}")

    # Prepare dataset
    dataset = NpyDataset(
        root=args.dataset_dir,
        folders=args.folders,
        transform=transform.Transforms(size=args.image_size, s=args.crop_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).test_transform,
        demo=args.demo
    )
    print("Dataset length:", len(dataset))

    batch_size = args.batch_size
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    for alpha in args.sparsity_alpha:
        for stickbreaking_strength in args.stickbreaking_strength:
            print(f"alpha: {alpha}, lambda: {stickbreaking_strength}")

            model_path = args.model_path.replace("EPOCH", str(args.epochs)).replace(
                "ALPHA", str(alpha)).replace("LAMBDA", str(stickbreaking_strength)).replace("CLASS", str(args.class_num))

            if not args.ps:
                model_path += "_nops"  # Append `_nops` for no participant supervision

            if args.demo:
                model_path += "_demo"

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            # Save the label mapping
            with open(os.path.join(model_path, 'label_mapping.pkl'), 'wb') as f:
                pickle.dump(dataset.label_mapping, f)

            # Load model based on `ps`
            res = resnet34.get_resnet(state_dict_path=None)
            if args.ps:
                model = network_stick.Network(res, args.feature_dim, args.class_num, n_participant, alpha)
            else:
                model = network_stick_nops.Network(res, args.feature_dim, args.class_num, n_participant, alpha)

            model_fp = os.path.join(model_path, "checkpoint.tar")
            model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
            model.to(device)

            print("### Creating features from model ###")
            results = inference(data_loader, model, device, args.ps)

            # Save cluster assignments
            df_columns = ['image_path', 'cluster_assignment', 'group_label']
            df = pd.DataFrame(columns=df_columns)

            if args.ps:
                cluster_assignment, group_assignment = results
                df["group_prediction"] = None  # Add group prediction column
            else:
                cluster_assignment = results  # No participant labels

            # Populate DataFrame
            for i in range(len(dataset)):
                image_path, group_label = dataset.get_image_path(i)
                cluster_id = cluster_assignment[i]
                row_data = {'image_path': image_path, 'cluster_assignment': cluster_id, 'group_label': group_label}
                
                if args.ps:
                    row_data['group_prediction'] = group_assignment[i]

                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

            print("Number of unique clusters:", df.cluster_assignment.nunique())

            # Save DataFrame
            df.to_csv(os.path.join(model_path, 'cluster_assignments.csv'), index=False)

            # Save results to a local directory
            local_result_dir = os.path.join(
                                            args.result_dir, 
                                            f"inference_pic_E{args.epochs}_A{alpha}_L{stickbreaking_strength}_C{args.class_num}"
                                        )                               
            if not args.ps:
                local_result_dir += "_nops"

            if args.demo:
                local_result_dir += "_demo"

            if not os.path.exists(local_result_dir):
                os.makedirs(local_result_dir)

            local_path = os.path.join(local_result_dir, "cluster_assignments.csv")
            subprocess.run(f"cp {os.path.join(model_path, 'cluster_assignments.csv')} {local_path}", shell=True)

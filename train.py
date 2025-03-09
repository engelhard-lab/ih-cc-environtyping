import os
import numpy as np
import torch
import argparse
from modules import transform, resnet34, network_stick, network_stick_nops, contrastive_loss
from modules.NpyDataset_stick import NpyDataset
from utils import yaml_config_hook
import torch.nn as nn


def check_nan_gradients(model):
    for param in model.parameters():
        if torch.isnan(param.grad).any():
            return True
    return False

def train(stickbreaking_strength):
    loss_epoch = 0
    for step, ((x_i, x_j), label) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda') 
        x_j = x_j.to('cuda')
        label = label.to('cuda')

        # Conditional model output based on `ps`
        if args.ps:
            z_i, z_j, c_i, c_j, s_i, s_j, p_i, p_j = model(x_i, x_j)
        else:
            z_i, z_j, c_i, c_j, s_i, s_j = model(x_i, x_j)

        # Compute losses
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss_stickbreaking = - (s_i + s_j) / 2 * stickbreaking_strength  

        # Add participant loss only if `ps=True`
        if args.ps:
            loss_participant = (criterion_participant(p_i, label) + criterion_participant(p_j, label)) / 2
            loss = loss_instance + loss_cluster + loss_participant + loss_stickbreaking
        else:
            loss = loss_instance + loss_cluster + loss_stickbreaking

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

        if step % 10 == 0:
            print(f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}\t loss_stickbreaking: {loss_stickbreaking.item()}")
            if args.ps:
                print(f"\t\tloss_participant: {loss_participant.item()}")

        loss_epoch += loss.item()
        optimizer.step()

    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to the configuration file")
    args, unknown = parser.parse_known_args()
    
    config = yaml_config_hook.yaml_config_hook(args.config)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.demo:
        ls = [name for name in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, name))]
        n_participant = len(ls)

    else:
        if not args.folders:
            args.folders = [int(name[13:-1]) for name in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, name))]
            n_participant = len(args.folders)
        else:
            n_participant = len(args.folders)

    print(f"Number of participant in the dataset: {n_participant}")
    print(f"Cluster head size: {args.class_num}")

    # Prepare dataset
    dataset = NpyDataset(
        root=args.dataset_dir,
        folders=args.folders,
        transform=transform.Transforms(size=args.image_size, s=args.crop_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        demo=args.demo
    ) 
    print("length of dataset", len(dataset))
    
    batch_size = args.batch_size

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    
    for alpha in args.sparsity_alpha:
        for stickbreaking_strength in args.stickbreaking_strength:
            print(f"alpha: {alpha}, lambda: {stickbreaking_strength}")
            
            model_path = args.model_path.replace("EPOCH", str(args.epochs)).replace(\
                "ALPHA", str(alpha)).replace(\
                    "LAMBDA", str(stickbreaking_strength)).replace("CLASS", str(args.class_num))
            
            # Append "_nops" if `ps=False`
            if not args.ps:
                model_path = model_path + "_nops"
            if args.demo:
                model_path = model_path + "_demo"
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            # Initialize model based on `ps`
            if args.reload:
                res = resnet34.get_resnet()
                model_class = network_stick.Network if args.ps else network_stick_nops.Network
                model = model_class(res, args.feature_dim, args.class_num, n_participant, alpha)
                model_fp = os.path.join(model_path, "checkpoint.tar")
                model.load_state_dict(torch.load(model_fp)['net'])
                model = model.to('cuda')
            else:
                res = resnet34.get_resnet(args.state_dict_path)
                model_class = network_stick.Network if args.ps else network_stick_nops.Network
                model = model_class(res, args.feature_dim, args.class_num, n_participant, alpha)
                model = model.to('cuda')

            # Optimizer & Loss
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
            loss_device = torch.device("cuda")

            # Loss functions
            criterion_instance = contrastive_loss.InstanceLoss(batch_size, args.instance_temperature, loss_device).to(loss_device)
            criterion_cluster = contrastive_loss.ClusterLoss(args.class_num, args.cluster_temperature, loss_device).to(loss_device)
            
            # Add participant loss criterion only if `ps=True`
            if args.ps:
                criterion_participant = nn.NLLLoss().to(loss_device)

            # Start training
            print(f"Start training, alpha = {alpha}, lambda = {stickbreaking_strength}, ps = {args.ps}")
          
            for epoch in range(args.start_epoch, args.epochs):
                lr = optimizer.param_groups[0]["lr"]
                loss_epoch = train(stickbreaking_strength)
                
                if epoch % 5 == 0:
                    print("Save model.....")
                    out = os.path.join(model_path, f"checkpoint_{epoch}.tar") 
                    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': args.epochs}
                    torch.save(state, out)

                print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}\t Current Learning Rate: {lr}")
                scheduler.step()
                print()

            # Save final model
            print("Save fully trained model.....")
            out = os.path.join(model_path, f"checkpoint.tar") 
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': args.epochs}
            torch.save(state, out)

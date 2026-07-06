import time
import os
import torch
import pandas as pd
import numpy as np


layout_to_n_containers = {
    (1,16,6)    :70, #(bay,row,tier)->container
    (2,16,6)    :140,
    (4,16,6)    :280,
    (6,16,6)    :430,
    (8,16,6)    :570,
    (10,16,6)   :720,
    (1,16,8)    :90,
    (2,16,8)    :190,
    (4,16,8)    :380,
    (6,16,8)    :570
}

def get_n_containers(n_bays, n_rows, n_tiers):
    return layout_to_n_containers[n_bays, n_rows, n_tiers]


def solve_benchmarks(model, epoch, args, instance_types):
    clock = time.time()
    model.eval()
    model.decoder.set_sampler('greedy')

    bays = [1,2,4,6,8,10]
    rows = [16]
    tiers = [6,8]
    data_path = './benchmarks/Lee_instances'

    for inst_type in instance_types:
        idxs = range(1,6) if inst_type == 'random' else range(1,3)

        data_names = []
        wts = {}

        for tier in tiers:
            for row in rows:
                for bay in bays:
                    if tier == 8 and bay in [8, 10]:
                        continue

                    inputs, names = zip(*[
                        find_and_process_file(data_path, inst_type, bay, row, tier, idx, no_print=True)
                        for idx in idxs
                    ])
                    inputs = torch.cat(inputs)

                    with torch.no_grad():
                        wt = model(inputs.to(args.device), None)
                        if isinstance(wt, tuple):
                            wt = wt[0]
                    
                    name = names[0][:-8]
                    data_names.append(name)
                    wts[name] = wt.mean().item()

        if inst_type == 'random':
            file_name = args.log_path + '/benchmark_WT(R).xlsx'
        else:
            file_name = args.log_path + '/benchmark_WT(U).xlsx'
        
        if not os.path.exists(file_name):
            df = pd.DataFrame(index=data_names)
            df.to_excel(file_name)

        df = pd.read_excel(file_name, index_col=0)
        df[f'Epoch {epoch+1}'] = df.index.map(wts)
        df.to_excel(file_name)

    print(f'Benchmark scoring time: {round(time.time() - clock, 1)}s')


def find_and_process_file(folder_path, inst_type, n_bays, n_rows, n_tiers, target_id, no_print=False):
    # Build the search pattern based on inputs
    stacks_str = f"{n_rows:02d}"  # e.g., 16 stacks -> "16"
    tiers_str = f"{n_tiers:02d}"  # e.g., 6 tiers -> "06"
    id_str = f"{target_id:03d}"  # e.g., id 3 -> "003"

    if inst_type == "random":
        folder_path += "/individual, random"
        bays_str = f"R{n_bays:02d}"  # e.g., 2 bays -> "R02"
    elif inst_type == "upsidedown":
        folder_path += "/individual, upside down"
        bays_str = f"U{n_bays:02d}"  # e.g., 2 bays -> "U02"
    else:
        raise ValueError(f"No instance type of '{inst_type}'")


    # Look for the matching file
    matching_files = [
        f for f in os.listdir(folder_path) 
        if f.startswith(f"{bays_str}{stacks_str}{tiers_str}") and f.endswith(f"_{id_str}.txt")
    ]
    
    if not matching_files:
        raise FileNotFoundError(f"No file found matching the criteria: bays={n_bays}, stacks={n_rows}, tiers={n_tiers}, id={target_id}")
    elif len(matching_files) > 1:
        raise ValueError(f"There are more than one file: {matching_files}")
    
    # Assuming the first match is the target file
    target_file = matching_files[0]
    if not no_print:
        print(f"Processing file: {target_file}")
    
    file_path = os.path.join(folder_path, target_file)

    return parse_container_file(file_path, n_bays, n_rows, n_tiers), target_file

def parse_container_file(file_path, n_bays, n_rows, n_tiers):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Create a zero-filled numpy array of shape (n_bays * n_rows, n_tiers)
    container_matrix = np.zeros((n_bays * n_rows, n_tiers), dtype=int)
    
    # Process each subsequent line
    for line in lines[1:]:  # Skip the first line
        values = list(map(int, line.split()))
        bay, stack, num_tiers = values[:3]
        container_numbers = values[3:]
        
        # Remove duplicates (since IDs are repeated twice)
        unique_container_numbers = list(dict.fromkeys(container_numbers))  # Preserve order while removing duplicates
        
        if len(unique_container_numbers) != num_tiers:
            ValueError(f'len(unique_container_numbers)(={len(unique_container_numbers)})'
                       +f' != numtiers(={num_tiers})')
        if len(unique_container_numbers) > n_tiers:
            ValueError(f'len(unique_container_numbers)(={len(unique_container_numbers)})'
                       +f' > n_tiers(={n_tiers})')
        
        # Pad with zeros if the number of unique containers is less than max tiers
        padded_containers = unique_container_numbers + [0] * (n_tiers - len(unique_container_numbers))
        
        # Compute the index in the matrix
        stack_index = (bay - 1) * n_rows + (stack - 1)
        
        # Fill the matrix with container numbers (bottom-up stacking)
        container_matrix[stack_index] = padded_containers
    
    # Convert to torch tensor with shape (1, n_bays * n_rows, n_tiers)
    container_tensor = torch.tensor(container_matrix).unsqueeze(0).float()
    return container_tensor.reshape(container_tensor.shape[0], n_bays, n_rows, container_tensor.shape[-1])

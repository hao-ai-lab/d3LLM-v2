import subprocess
import json
import os
from datasets import Dataset
from datasets import load_dataset
import argparse


def main(
    num_gpus=24,
    steps=256,
    gen_length=256,
    block_length=32,
    output_dir="trajectory_output",
    max_data_num=-1,
):
    """Distributed trajectory generation for DREAM using multiple GPUs across multiple nodes"""

    # Get SLURM task info
    # SLURM_PROCID: global rank (0 to num_gpus-1)
    # SLURM_LOCALID: local GPU ID on this node (0-3)
    # SLURM_NTASKS: total number of tasks (should equal num_gpus)
    
    slurm_procid = int(os.environ.get("SLURM_PROCID", "0"))
    slurm_localid = int(os.environ.get("SLURM_LOCALID", "0"))
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", str(num_gpus)))
    
    print(f"Task {slurm_procid}/{slurm_ntasks}, Local GPU {slurm_localid}")
    
    # Only the first task does dataset loading and final concatenation
    if slurm_procid == 0:
        # Load dataset to get total size
        dataset = load_dataset("Zigeng/dParallel_Dream_Distill_Data", split="train")
        # dataset = load_dataset("coder_data/Ling-Coder-dParallel-merged-512-120k", split="train")
        total_size = len(dataset)

        # Apply max_data_num limit
        if max_data_num > 0:
            total_size = min(total_size, max_data_num)

        os.makedirs(output_dir, exist_ok=True)
        
        # Save total_size to a file for other tasks
        with open(os.path.join(output_dir, "total_size.txt"), "w") as f:
            f.write(str(total_size))
        
        print(f"Total dataset size: {total_size}")
        print(f"Distributing across {num_gpus} GPUs")
    
    # Barrier: wait for task 0 to write total_size
    # In SLURM with srun, we can use a simple file-based barrier
    import time
    total_size_file = os.path.join(output_dir, "total_size.txt")
    while not os.path.exists(total_size_file):
        time.sleep(1)
    
    with open(total_size_file, "r") as f:
        total_size = int(f.read().strip())
    
    # Calculate this task's chunk
    chunk_size = (total_size + num_gpus - 1) // num_gpus
    gpu_id = slurm_procid  # Use global rank as gpu_id
    start_idx = gpu_id * chunk_size
    end_idx = min((gpu_id + 1) * chunk_size, total_size)
    output_file = os.path.join(output_dir, f"trajectory_part_{gpu_id}.json")

    # Run generation on this GPU
    cmd = [
        "python",
        "d3llm/d3llm_DREAM/distill_1_data_prepare/d3llm_dream_generate_partly.py",
        "--start_idx",
        str(start_idx),
        "--end_idx",
        str(end_idx),
        "--steps",
        str(steps),
        "--gen_length",
        str(gen_length),
        "--block_length",
        str(block_length),
        "--output_file",
        output_file,
        "--max_data_num",
        str(max_data_num),
    ]

    env = os.environ.copy()
    # Use local GPU ID
    env["CUDA_VISIBLE_DEVICES"] = str(slurm_localid)

    print(f"GPU {gpu_id}: Processing indices {start_idx}-{end_idx}")
    result = subprocess.run(cmd, env=env)
    
    if result.returncode != 0:
        print(f"GPU {gpu_id}: Generation failed with return code {result.returncode}")
        return

    print(f"GPU {gpu_id}: Generation completed")

    # Barrier: wait for all tasks to complete
    # Create a completion flag for this task
    completion_file = os.path.join(output_dir, f"completed_{gpu_id}.flag")
    with open(completion_file, "w") as f:
        f.write("done")
    
    # Only task 0 does concatenation
    if slurm_procid == 0:
        print("Waiting for all tasks to complete...")
        # Wait for all completion flags
        while True:
            completed = sum(
                1 for i in range(num_gpus)
                if os.path.exists(os.path.join(output_dir, f"completed_{i}.flag"))
            )
            if completed == num_gpus:
                break
            print(f"Completed: {completed}/{num_gpus}")
            time.sleep(5)
        
        print("All tasks completed. Concatenating results...")
        
        # Concatenate results
        all_data = []
        for gpu_id in range(num_gpus):
            part_file = os.path.join(output_dir, f"trajectory_part_{gpu_id}.json")
            if os.path.exists(part_file):
                with open(part_file, "r") as f:
                    data = json.load(f)
                    all_data.extend(data)
                    print(f"Loaded {len(data)} samples from GPU {gpu_id}")
            else:
                print(f"Warning: {part_file} not found")

        # Convert to dataset format with correctness check
        dataset_dict = {
            "idx": [d["idx"] for d in all_data],
            "question": [d["question"] for d in all_data],
            "prompt_ids": [d["prompt_ids"] for d in all_data],
            "trajectory": [d["trajectory"] for d in all_data],
            "final_output": [d["final_output"] for d in all_data],
            "generated_text": [d["generated_text"] for d in all_data],
            "llm_answer": [d["llm_answer"] for d in all_data],
            "gt_answer": [d["gt_answer"] for d in all_data],
            "is_correct": [d["is_correct"] for d in all_data],
            "nfe": [d.get("nfe", 0) for d in all_data],  # Add NFE field
        }

        # Print statistics
        num_correct = sum(dataset_dict["is_correct"])
        total = len(dataset_dict["is_correct"])
        accuracy = num_correct / total if total > 0 else 0
        avg_nfe = sum(dataset_dict["nfe"]) / total if total > 0 else 0
        print(f"Correctness: {num_correct}/{total} = {accuracy:.2%}")
        print(f"Average NFE: {avg_nfe:.2f}")

        final_dataset = Dataset.from_dict(dataset_dict)
        final_dataset.save_to_disk(os.path.join(output_dir, "trajectory_dataset"))
        print(
            f"Saved complete dataset with {len(all_data)} samples to {output_dir}/trajectory_dataset"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=24)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="trajectory_output")
    parser.add_argument(
        "--max_data_num",
        type=int,
        default=-1,
        help="Max number of samples to generate (-1 for no limit)",
    )
    args = parser.parse_args()

    main(
        args.num_gpus,
        args.steps,
        args.gen_length,
        args.block_length,
        args.output_dir,
        args.max_data_num,
    )

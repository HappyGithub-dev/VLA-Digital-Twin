import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tensorflow_datasets as tfds
from tqdm import tqdm
import os

# Import your local classes
from models.SmolLM2 import SmolVLA, apply_compression


# --- 1. FIXED RLDS LOADING WITH RESIZING ---
def get_dataloader(batch_size=1):
    # Standard path to your data folder
    data_dir = "/data"

    print(f"✅ Loading RLDS dataset and resizing to 224x224 from: {data_dir}")

    # Load local RLDS dataset
    builder = tfds.builder_from_directory(
        os.path.join(data_dir, "libero_spatial_no_noops", "1.0.0")
    )
    ds = builder.as_dataset(split='train')

    def generator():
        for episode in ds:
            for step in episode['steps']:
                # 1. Convert to Tensor
                img_raw = step['observation']['image'].numpy()
                img = torch.from_numpy(img_raw).permute(2, 0, 1).float() / 255.0

                # 2. THE FIX: Force Resize to 224x224
                # We use bilinear interpolation to maintain image quality
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                # State processing (Pad to 32-dim)
                state_raw = torch.from_numpy(step['observation']['state'].numpy()).float()
                state = torch.zeros(32)
                state[:state_raw.shape[0]] = state_raw

                # Action vector
                action = torch.from_numpy(step['action'].numpy()).float()

                yield {"img": img, "state": state, "action": action}

    class RobotDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            return generator()

    return DataLoader(RobotDataset(), batch_size=batch_size)


# --- 2. DISTILLATION ENGINE ---
# --- 2. DISTILLATION ENGINE (WITH 10,000 STEP LIMIT) ---
def run_distillation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Init] Training on: {device}")

    # Initialize Models
    teacher = SmolVLA().to(device).to(torch.bfloat16).eval()
    student = SmolVLA().to(device).to(torch.bfloat16)

    # Apply your compression logic
    apply_compression(student)
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    dataloader = get_dataloader(batch_size=1)

    task_text = "pick up the black bowl"
    tokens = student.tokenizer(task_text, return_tensors="pt").input_ids.to(device)

    # --- STEP LIMITER SETUP ---
    max_steps = 10000
    current_step = 0

    print(f"\n🚀 Starting Distillation (Limited to {max_steps} steps)...")

    loop = tqdm(dataloader, desc="Training")
    for batch in loop:
        if current_step >= max_steps:
            break  # Exit the loop once we hit 10,000 steps

        imgs = batch['img'].to(device).to(torch.bfloat16)
        states = batch['state'].to(device).to(torch.bfloat16)

        with torch.no_grad():
            t_act, t_feat = teacher(imgs, tokens, states)

        s_act, s_feat = student(imgs, tokens, states)

        loss_act = F.mse_loss(s_act, t_act)
        loss_feat = 1 - F.cosine_similarity(s_feat, t_feat, dim=-1).mean()

        total_loss = (15.0 * loss_act) + (2.0 * loss_feat)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        current_step += 1
        loop.set_postfix(
            step=f"{current_step}/{max_steps}",
            loss=f"{total_loss.item():.4f}",
            align=f"{(1 - loss_feat.item()):.1%}"
        )

    # Save the model so we don't lose progress
    torch.save(student.state_dict(), "../SmolVLA_104M_Distilled.pth")
    print(f"\n✅ Milestone Reached! Model saved after {current_step} steps.")


if __name__ == "__main__":
    run_distillation()
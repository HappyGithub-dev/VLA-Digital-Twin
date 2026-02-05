import torch
import torch.nn.functional as F
import numpy as np
import robosuite
import cv2  # Required to save the video file
import os
import sys

# 1. PATH FIX: Ensure the 'models' folder is findable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from models.SmolLM2 import SmolVLA, apply_compression


def run_simulation():
    # Use MPS for Apple Silicon performance
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Sim] Digital Twin Active on: {device}")

    # 2. LOAD THE DISTILLED BRAIN
    model = SmolVLA().to(device).to(torch.bfloat16)
    apply_compression(model)
    model_path = os.path.join(BASE_DIR, "SmolVLA_104M_Distilled.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. SETUP ENV (Offscreen Only to prevent Mac crashes)
    # Removed camera_height/width to resolve your TypeError
    env = robosuite.make(
        env_name="PickPlace",
        robots="Panda",
        has_renderer=False,  # Disable the pop-up window to avoid crashes
        has_offscreen_renderer=True,  # Enable internal camera rendering
        use_camera_obs=True,
        camera_names="agentview",
    )

    # 4. VIDEO RECORDER SETUP
    video_path = os.path.join(BASE_DIR, "digital_twin_demo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Use standard 512x512 for the demo video
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (512, 512))

    obs = env.reset()
    task_text = "pick up the black bowl"
    tokens = model.tokenizer(task_text, return_tensors="pt").input_ids.to(device)

    print(f"🚀 Simulation Running... recording to: {video_path}")

    try:
        for step in range(500):  # 500 steps captures the full movement
            # A. PREPROCESS IMAGE FOR VIDEO & MODEL
            img_raw = obs['agentview_image']

            # Save frame to video (Need to flip because MuJoCo renders upside down)
            frame = np.flipud(img_raw)
            # Resize for the video file output
            frame_resized = cv2.resize(frame, (512, 512))
            video_writer.write(cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR))

            # Resize specifically for the SigLIP vision tower (224x224)
            img_tensor = torch.from_numpy(img_raw).permute(2, 0, 1).float() / 255.0
            img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(224, 224), mode='bilinear')
            img_tensor = img_tensor.to(device).to(torch.bfloat16)

            # B. FIX STATE DIMENSION (43 -> 32)
            state_raw = torch.from_numpy(obs['robot0_proprio-state']).float()
            state = torch.zeros(1, 32).to(device).to(torch.bfloat16)
            state[0, :] = state_raw[:32]

            # C. MODEL INFERENCE
            with torch.no_grad():
                action, _ = model(img_tensor, tokens, state)

            # D. EXECUTE STEP
            env_action = action.squeeze().cpu().float().numpy()
            obs, reward, done, info = env.step(env_action)

            if step % 100 == 0:
                print(f"Step {step}/500 processed...")

            if done:
                print("🏁 Goal Achieved!")
                break

    finally:
        video_writer.release()
        env.close()
        print(f"🎬 Done! Video saved at {video_path}")


if __name__ == "__main__":
    run_simulation()
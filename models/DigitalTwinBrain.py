import torch
from models.SmolLM2 import SmolVLA, apply_compression

class DigitalTwinBrain:
    def __init__(self, checkpoint_path="SmolVLA_104M_Distilled.pth"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SmolVLA().to(self.device).to(torch.bfloat16)
        apply_compression(self.model) # Essential: matches training surgery
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def predict(self, image_224, task_tokens, state_32):
        # Wraps the inference logic
        with torch.no_grad():
            action, _ = self.model(image_224, task_tokens, state_32)
        return action.squeeze().cpu().float().numpy()
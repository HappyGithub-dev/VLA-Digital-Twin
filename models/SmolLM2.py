import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

# 1. SETUP
LLM_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SIGLIP_ID = "google/siglip-base-patch16-224"


# --- CKA MATHEMATICAL ENGINE ---
def hsic_unbiased(K, L):
    """Calculates unbiased HSIC with dimension safety."""
    n = K.shape[0]
    if n < 2:
        # CKA requires at least 2 samples to calculate 'unbiased' variance
        # Since we use a single calibration frame, we fallback to biased HSIC
        K_c = K - K.mean()
        L_c = L - L.mean()
        return torch.trace(K_c @ L_c)

    H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
    K_c, L_c = H @ K @ H, H @ L @ H
    return torch.trace(K_c @ L_c) / ((n - 1) ** 2)


def calculate_cka(feat_a, feat_b):
    """Computes similarity between layer activations with proper reshaping."""
    # THE FIX: Ensure input is 2D (samples, features)
    # feat_a/b are currently [1, Hidden_Dim] from the mean pooling
    # We use .view(feat_a.size(0), -1) to be safe
    a = feat_a.view(feat_a.size(0), -1)
    b = feat_b.view(feat_b.size(0), -1)

    # Modern Transpose (.mT) to fix UserWarning and handle batches
    K = a @ a.mT
    L = b @ b.mT

    hsic_kl = hsic_unbiased(K, L)
    hsic_kk = hsic_unbiased(K, K)
    hsic_ll = hsic_unbiased(L, L)
    return hsic_kl / (torch.sqrt(hsic_kk) * torch.sqrt(hsic_ll))


class FeatureExtractor:
    def __init__(self):
        self.activations = []

    def hook_fn(self, module, input, output):
        # Pool spatial dimensions to reduce memory
        self.activations.append(output[0].detach().mean(dim=1).float())


class SmolVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # VISION TOWER
        self.vision_tower = SiglipVisionModel.from_pretrained(SIGLIP_ID, dtype=torch.bfloat16).to(self.device)

        # LLM BACKBONE
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(LLM_ID, dtype=torch.bfloat16, trust_remote_code=True).to(
            self.device)

        # CHANNELS
        h_dim = self.llm.config.hidden_size
        self.vision_projector = nn.Linear(768, h_dim).to(torch.bfloat16).to(self.device)
        self.state_mlp = nn.Sequential(nn.Linear(32, h_dim)).to(torch.bfloat16).to(self.device)
        self.action_head = nn.Linear(h_dim, 7).to(torch.bfloat16).to(self.device)

    def forward(self, image, text_ids, state):
        vis_tokens = self.vision_tower(image).last_hidden_state
        vis_embeds = self.vision_projector(vis_tokens)

        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)

        text_embeds = self.llm.get_input_embeddings()(text_ids)
        state_embeds = self.state_mlp(state).unsqueeze(1)

        inputs = torch.cat([text_embeds, vis_embeds, state_embeds], dim=1)
        outputs = self.llm(inputs_embeds=inputs, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1][:, -1, :]
        action = torch.tanh(self.action_head(last_hidden))  # Added Tanh for robotic normalization

        return action, last_hidden

    def get_cka_activations(self, image, text_ids, state):
        """Captures activations from all 12 vision layers."""
        extractor = FeatureExtractor()
        hooks = [l.register_forward_hook(extractor.hook_fn) for l in self.vision_tower.vision_model.encoder.layers]
        with torch.no_grad(): self.forward(image, text_ids, state)
        for h in hooks: h.remove()
        return extractor.activations

    def audit(self, label="REPORT"):
        v_params = sum(p.numel() for p in self.vision_tower.parameters()) / 1e6
        v_layers = len(self.vision_tower.vision_model.encoder.layers)
        l_params = sum(p.numel() for p in self.llm.parameters()) / 1e6
        l_layers = len(self.llm.model.layers)
        s_params = (sum(p.numel() for p in self.state_mlp.parameters()) +
                    sum(p.numel() for p in self.vision_projector.parameters())) / 1e6
        total = v_params + l_params + s_params + (sum(p.numel() for p in self.action_head.parameters()) / 1e6)

        print(f"\n{'=' * 10} {label} {'=' * 10}")
        print(f"1) Vision Tower:  {v_params:>6.2f}M | Layers: {v_layers}")
        print(f"2) Smol LM2:      {l_params:>6.2f}M | Layers: {l_layers}")
        print(f"3) State Encoder: {s_params:>6.2f}M")
        print(f"{'-' * 32}")
        print(f"TOTAL PARAMETERS: {total:.2f}M")
        print(f"{'=' * 30}\n")


def apply_compression(model, calib_image=None, calib_text=None, calib_state=None):
    # --- CKA ANALYSIS FOR VISION TOWER ---
    if calib_image is not None and len(model.vision_tower.vision_model.encoder.layers) > 5:
        print("[*] Performing CKA Analysis on Vision Tower...")
        acts = model.get_cka_activations(calib_image, calib_text, calib_state)
        # Calculate 'Semantic Shifts' (1 - Similarity)
        diffs = [1.0 - calculate_cka(acts[i], acts[i + 1]).item() for i in range(len(acts) - 1)]

        # Keep Input (0), Output (11), and 3 most unique middle layers
        important_middles = np.argsort(diffs)[-3:] + 1
        v_indices = sorted([0] + important_middles.tolist() + [11])

        print(f"[*] CKA Selected Vision Indices: {v_indices}")
        model.vision_tower.vision_model.encoder.layers = nn.ModuleList([
            model.vision_tower.vision_model.encoder.layers[i] for i in v_indices
        ])

    # --- STRIDED PRUNING FOR LLM ---
    if len(model.llm.model.layers) > 15:
        # Uniformly sample across the 30 layers to maintain reasoning depth
        l_indices = np.linspace(0, len(model.llm.model.layers) - 1, 15, dtype=int).tolist()
        model.llm.model.layers = nn.ModuleList([model.llm.model.layers[i] for i in l_indices])

    # --- SVD COMPRESSION ---
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and "llm" in name and "action_head" not in name:
            rank = max(1, int(min(module.in_features, module.out_features) * 0.3))
            with torch.no_grad():
                W = module.weight.data.float().cpu()
                U, S, V = torch.linalg.svd(W, full_matrices=False)
                new_m = nn.Sequential(
                    nn.Linear(module.in_features, rank, bias=False),
                    nn.Linear(rank, module.out_features, bias=(module.bias is not None))
                ).to(torch.bfloat16).to(model.llm.device)

                new_m[0].weight.data = (torch.diag(torch.sqrt(S[:rank])) @ V[:rank, :]).to(torch.bfloat16).to(
                    model.llm.device)
                new_m[1].weight.data = (U[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))).to(torch.bfloat16).to(
                    model.llm.device)

                parent = model
                parts = name.split('.')
                for part in parts[:-1]: parent = getattr(parent, part)
                setattr(parent, parts[-1], new_m)


if __name__ == "__main__":
    vla = SmolVLA()
    vla.audit("BEFORE COMPRESSION")

    # DUMMIES FOR CKA CALIBRATION
    img = torch.randn(1, 3, 224, 224).to(vla.device).to(torch.bfloat16)
    txt = torch.randint(0, 1000, (1, 16)).to(vla.device)
    st = torch.randn(1, 32).to(vla.device).to(torch.bfloat16)

    apply_compression(vla, img, txt, st)
    vla.audit("AFTER COMPRESSION")
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

# 1. SETUP

LLM_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SIGLIP_ID = "google/siglip-base-patch16-224"


class SmolVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # VISION TOWER - Updated torch_dtype to dtype
        self.vision_tower = SiglipVisionModel.from_pretrained(SIGLIP_ID, dtype=torch.bfloat16).to(self.device)

        # LLM BACKBONE - Updated torch_dtype to dtype
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(LLM_ID, dtype=torch.bfloat16, trust_remote_code=True).to(
            self.device)

        # CHANNELS
        h_dim = self.llm.config.hidden_size  # 576 for SmolLM2-135M
        self.vision_projector = nn.Linear(768, h_dim).to(torch.bfloat16).to(self.device)
        self.state_mlp = nn.Sequential(nn.Linear(32, h_dim)).to(torch.bfloat16).to(self.device)
        self.action_head = nn.Linear(h_dim, 7).to(torch.bfloat16).to(self.device)

    # --- THE MISSING FORWARD FUNCTION ---
    def forward(self, image, text_ids, state):
        # 1. Vision: SigLIP always gives 196 tokens [B, 196, 768]
        vis_tokens = self.vision_tower(image).last_hidden_state
        vis_embeds = self.vision_projector(vis_tokens)  # [B, 196, 576]

        # 2. Text: Ensure text_ids is 2D [B, L]
        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)

        text_embeds = self.llm.get_input_embeddings()(text_ids)  # [B, L, 576]

        # 3. State: [B, 32] -> [B, 1, 576]
        state_embeds = self.state_mlp(state).unsqueeze(1)

        # --- THE FIX: Concatenation without shape assumptions ---
        # We concatenate everything along the sequence dimension (dim 1)
        # Total tokens = L + 196 + 1
        inputs = torch.cat([text_embeds, vis_embeds, state_embeds], dim=1)

        # 4. LLM Forward Pass
        # We pass only inputs_embeds to avoid the LLM trying to use its own tokenizer
        outputs = self.llm(inputs_embeds=inputs, output_hidden_states=True)

        # 5. Feature Extraction
        # We take only the LAST hidden state for the entire sequence.
        # This is a [B, 576] vector representing the "final decision".
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        # 6. Action Prediction
        action = self.action_head(last_hidden)

        return action, last_hidden

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


def apply_compression(model):
    # Prune Vision Tower to 5 layers
    if len(model.vision_tower.vision_model.encoder.layers) > 5:
        v_indices = [0, 2, 5, 8, 11]
        model.vision_tower.vision_model.encoder.layers = nn.ModuleList([
            model.vision_tower.vision_model.encoder.layers[i] for i in v_indices
        ])

    # Prune LLM to 15 layers
    if len(model.llm.model.layers) > 15:
        model.llm.model.layers = nn.ModuleList([model.llm.model.layers[i] for i in range(15)])

    # SVD Compression
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
                parent = model;
                parts = name.split('.')
                for part in parts[:-1]: parent = getattr(parent, part)
                setattr(parent, parts[-1], new_m)


if __name__ == "__main__":
    vla = SmolVLA()
    vla.audit("BEFORE COMPRESSION")
    apply_compression(vla)
    vla.audit("AFTER COMPRESSION")
# VLA-Digital-Twin
This Repository Contains Files related to VLM backbone. 

The objective of the project was to develop a digital twin of the VLA Model that could perform simulations on different GPUs. The task assigned to was to select a pair of LLM, Vision Encoder and State Encoder and build a model that could take a camera feed , text input , and the robot gripper state and generate the action sequence for the robot to complete the given task. 

The Model seleted was - SmolLM2 for the LLM backbone, SigLIP as the vision encoder.

The model was compressed by using techniques like - pruning, low rank factorization and student distillation. 

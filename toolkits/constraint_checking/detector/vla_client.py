"""OpenVLA-7B inference wrapper."""
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class OpenVLAClient:
    """
    Action format: [dx, dy, dz, droll, dpitch, dyaw, gripper].
    Gripper convention depends on unnorm_key:
      - bridge_orig: ~0 = close, ~1 = open
      - libero_object: ~0 = open, ~1 = close
    """

    def __init__(
        self,
        model_id: str = 'openvla/openvla-7b',
        device: str = 'cuda:0',
        unnorm_key: str = 'bridge_orig',
        dtype=torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.unnorm_key = unnorm_key

        print(f'[VLA] loading processor: {model_id}')
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print(f'[VLA] loading model: {model_id} (first run downloads ~16GB)')
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        print('[VLA] ready.')

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, instruction: str) -> np.ndarray:
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        prompt = f'In: What action should the robot take to {instruction.strip().lower()}?\nOut:'
        inputs = self.processor(prompt, img).to(self.device, dtype=self.dtype)
        action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return np.asarray(action, dtype=np.float32)

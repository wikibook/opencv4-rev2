import torch
from transformer_net import TransformerNet

for name in ["mosaic", "candy", "rain_princess", "udnie"]:
    model = TransformerNet()
    state_dict = torch.load(f"./pytorch_model/{name}.pth")

    for key in list(state_dict.keys()):
        if key.endswith(("running_mean", "running_var")):
            del state_dict[key]

    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(model, dummy_input, f"./onnx_model/{name}.onnx")
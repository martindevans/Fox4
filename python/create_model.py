#!/usr/bin/env python3

import torch
import onnx
import argparse

from model import Fox4Network

MODEL_VERSION = "0.2"


def create_random_model():
    """Create a random model"""
    
    print("🎲 Creating random Fox4 model...")
    
    model = Fox4Network(input_size=36)
    
    # Randomize weights
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.5)  # Smaller gain, previous was generating large values
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)  # Start with zero bias
    
    model.eval()
    
    dummy_input = torch.randn(1, 36)
    
    onnx_path = "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Add version metadata
    onnx_model = onnx.load(onnx_path)
    onnx_model.metadata_props.append(onnx.StringStringEntryProto(key="version", value=MODEL_VERSION))
    onnx.save(onnx_model, onnx_path)
    
    print(f"✅ Random model exported to: {onnx_path} with version {MODEL_VERSION}")
    
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Create ONNX model for Fox4')
    parser.add_argument('--random', action='store_true', help='Create random model for testing')
    
    args = parser.parse_args()
    
    if args.random:
        create_random_model()


if __name__ == "__main__":
    main()

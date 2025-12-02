#!/usr/bin/env python3
import torch
import numpy as np
import os
from pathlib import Path

class PyTorchToHLSConverter:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.weights = {}
        self.weight_shapes = {}  # Store shapes for header generation
        self.load_all_checkpoints()

    def load_all_checkpoints(self):
        """Load all PyTorch checkpoints and extract weights."""
        checkpoint_files = {
            'input_conv': 'input_conv.pth',
            'encoder_conv': 'encoder_conv.pth',
            'decoder_conv': 'decoder_conv.pth',
            'output_conv': 'output_conv.pth',
            'final_conv': 'final_conv.pth'
        }

        for block_name, filename in checkpoint_files.items():
            filepath = self.checkpoint_dir / filename
            if filepath.exists():
                checkpoint = torch.load(filepath, map_location='cpu')
                self.weights[block_name] = checkpoint
                print(f"Loaded {block_name} from {filename}")
            else:
                print(f"Warning: {filename} not found")

    def extract_conv_weights(self, state_dict, conv_name):
        """Extract convolution weights and bias from state_dict."""
        weight_key = f'{conv_name}.weight'
        bias_key = f'{conv_name}.bias'

        weight = state_dict.get(weight_key, None)
        bias = state_dict.get(bias_key, None)

        if weight is not None:
            weight = weight.detach().numpy()
        if bias is not None:
            bias = bias.detach().numpy()

        return weight, bias

    def extract_groupnorm_weights(self, state_dict, gn_name):
        """Extract GroupNorm gamma (weight) and beta (bias) from state_dict."""
        gamma_key = f'{gn_name}.weight'
        beta_key = f'{gn_name}.bias'

        gamma = state_dict.get(gamma_key, None)
        beta = state_dict.get(beta_key, None)

        if gamma is not None:
            gamma = gamma.detach().numpy()
        if beta is not None:
            beta = beta.detach().numpy()

        return gamma, beta

    def format_array_for_cpp(self, array, name, shape_comment=""):
        """Format numpy array as C++ array initialization with proper multi-dimensional syntax."""
        if array is None:
            return f"// {name} not found in checkpoint\n"

        # Store shape information for header generation
        self.weight_shapes[name] = array.shape

        flat_data = array.flatten()
        shape_str = ''.join(f'[{dim}]' for dim in array.shape)

        lines = []
        lines.append(f"// {name} {shape_comment}Shape: {array.shape}")
        lines.append(f"float {name}{shape_str} = {{")

        # Format values with 6 decimal places, 8 values per line
        for i in range(0, len(flat_data), 8):
            chunk = flat_data[i:i+8]
            formatted_chunk = [f"{val:.6f}f" for val in chunk]
            lines.append("    " + ", ".join(formatted_chunk) +
                        ("," if i + 8 < len(flat_data) else ""))

        lines.append("};")
        lines.append("")

        return "\n".join(lines)

    def generate_hls_weights(self, output_file="unet_weights_data.cpp"):
        """Generate C++ file with all extracted weights."""

        with open(output_file, 'w') as f:
            f.write("#include \"unet_weights.h\"\n\n")
            f.write("// Auto-generated weights from PyTorch checkpoints\n")
            f.write("// 3D U-Net with reduced contraction path\n\n")

            # Input convolution block (1->64->64)
            if 'input_conv' in self.weights:
                state_dict = self.weights['input_conv']['state_dict']

                # First conv layer
                conv1_weight, conv1_bias = self.extract_conv_weights(state_dict, 'conv_block.SingleConv1.conv')
                conv1_gamma, conv1_beta = self.extract_groupnorm_weights(state_dict, 'conv_block.SingleConv1.groupnorm')

                f.write("// === INPUT CONVOLUTION BLOCK ===\n")
                f.write(self.format_array_for_cpp(conv1_weight, "input_conv1_weight"))
                f.write(self.format_array_for_cpp(conv1_gamma, "input_conv1_gamma"))
                f.write(self.format_array_for_cpp(conv1_beta, "input_conv1_beta"))

                # Second conv layer
                conv2_weight, conv2_bias = self.extract_conv_weights(state_dict, 'conv_block.SingleConv2.conv')
                conv2_gamma, conv2_beta = self.extract_groupnorm_weights(state_dict, 'conv_block.SingleConv2.groupnorm')

                f.write(self.format_array_for_cpp(conv2_weight, "input_conv2_weight"))
                f.write(self.format_array_for_cpp(conv2_gamma, "input_conv2_gamma"))
                f.write(self.format_array_for_cpp(conv2_beta, "input_conv2_beta"))

            # Encoder convolution block (64->128->128)
            if 'encoder_conv' in self.weights:
                state_dict = self.weights['encoder_conv']['state_dict']

                # First conv layer
                conv1_weight, conv1_bias = self.extract_conv_weights(state_dict, 'conv_block.SingleConv1.conv')
                conv1_gamma, conv1_beta = self.extract_groupnorm_weights(state_dict, 'conv_block.SingleConv1.groupnorm')

                f.write("// === ENCODER CONVOLUTION BLOCK ===\n")
                f.write(self.format_array_for_cpp(conv1_weight, "encoder_conv1_weight"))
                f.write(self.format_array_for_cpp(conv1_gamma, "encoder_conv1_gamma"))
                f.write(self.format_array_for_cpp(conv1_beta, "encoder_conv1_beta"))

                # Second conv layer
                conv2_weight, conv2_bias = self.extract_conv_weights(state_dict, 'conv_block.SingleConv2.conv')
                conv2_gamma, conv2_beta = self.extract_groupnorm_weights(state_dict, 'conv_block.SingleConv2.groupnorm')

                f.write(self.format_array_for_cpp(conv2_weight, "encoder_conv2_weight"))
                f.write(self.format_array_for_cpp(conv2_gamma, "encoder_conv2_gamma"))
                f.write(self.format_array_for_cpp(conv2_beta, "encoder_conv2_beta"))

            # Decoder convolution block (192->64->64)
            if 'decoder_conv' in self.weights:
                state_dict = self.weights['decoder_conv']['state_dict']

                # First conv layer
                conv1_weight, conv1_bias = self.extract_conv_weights(state_dict, 'conv_block.SingleConv1.conv')
                conv1_gamma, conv1_beta = self.extract_groupnorm_weights(state_dict, 'conv_block.SingleConv1.groupnorm')

                f.write("// === DECODER CONVOLUTION BLOCK ===\n")
                f.write(self.format_array_for_cpp(conv1_weight, "decoder_conv1_weight"))
                f.write(self.format_array_for_cpp(conv1_gamma, "decoder_conv1_gamma"))
                f.write(self.format_array_for_cpp(conv1_beta, "decoder_conv1_beta"))

                # Second conv layer
                conv2_weight, conv2_bias = self.extract_conv_weights(state_dict, 'conv_block.SingleConv2.conv')
                conv2_gamma, conv2_beta = self.extract_groupnorm_weights(state_dict, 'conv_block.SingleConv2.groupnorm')

                f.write(self.format_array_for_cpp(conv2_weight, "decoder_conv2_weight"))
                f.write(self.format_array_for_cpp(conv2_gamma, "decoder_conv2_gamma"))
                f.write(self.format_array_for_cpp(conv2_beta, "decoder_conv2_beta"))

            # Output convolution block (64->64->64)
            if 'output_conv' in self.weights:
                state_dict = self.weights['output_conv']['state_dict']

                # First conv layer
                conv1_weight, conv1_bias = self.extract_conv_weights(state_dict, 'conv_block.SingleConv1.conv')
                conv1_gamma, conv1_beta = self.extract_groupnorm_weights(state_dict, 'conv_block.SingleConv1.groupnorm')

                f.write("// === OUTPUT CONVOLUTION BLOCK ===\n")
                f.write(self.format_array_for_cpp(conv1_weight, "output_conv1_weight"))
                f.write(self.format_array_for_cpp(conv1_gamma, "output_conv1_gamma"))
                f.write(self.format_array_for_cpp(conv1_beta, "output_conv1_beta"))

                # Second conv layer
                conv2_weight, conv2_bias = self.extract_conv_weights(state_dict, 'conv_block.SingleConv2.conv')
                conv2_gamma, conv2_beta = self.extract_groupnorm_weights(state_dict, 'conv_block.SingleConv2.groupnorm')

                f.write(self.format_array_for_cpp(conv2_weight, "output_conv2_weight"))
                f.write(self.format_array_for_cpp(conv2_gamma, "output_conv2_gamma"))
                f.write(self.format_array_for_cpp(conv2_beta, "output_conv2_beta"))

            # Final convolution (64->5)
            if 'final_conv' in self.weights:
                state_dict = self.weights['final_conv']['state_dict']

                final_weight = state_dict.get('weight', None)
                final_bias = state_dict.get('bias', None)

                if final_weight is not None:
                    final_weight = final_weight.detach().numpy()
                if final_bias is not None:
                    final_bias = final_bias.detach().numpy()

                f.write("// === FINAL CONVOLUTION ===\n")
                f.write(self.format_array_for_cpp(final_weight, "final_conv_weight"))
                f.write(self.format_array_for_cpp(final_bias, "final_conv_bias"))

        print(f"Generated {output_file} with extracted weights")

    def generate_weight_header(self, output_file="unet_weights.h"):
        """Generate C++ header file with weight declarations using actual shapes."""

        with open(output_file, 'w') as f:
            f.write("#ifndef UNET_WEIGHTS_H\n")
            f.write("#define UNET_WEIGHTS_H\n\n")
            f.write("// Auto-generated weight declarations for 3D U-Net\n")
            f.write("// Reduced contraction path architecture\n\n")

            # Helper function to write extern declaration with shape
            def write_extern_with_shape(var_name, comment=""):
                if var_name in self.weight_shapes:
                    shape = self.weight_shapes[var_name]
                    shape_str = ''.join(f'[{dim}]' for dim in shape)
                    f.write(f"extern float {var_name}{shape_str};\n")
                else:
                    f.write(f"extern float {var_name}[];  // Shape not available\n")

            # Input convolution weights (1->64->64)
            f.write("// Input Convolution Block: 1 -> 64 -> 64 channels\n")
            write_extern_with_shape("input_conv1_weight")
            write_extern_with_shape("input_conv1_gamma")
            write_extern_with_shape("input_conv1_beta")
            write_extern_with_shape("input_conv2_weight")
            write_extern_with_shape("input_conv2_gamma")
            write_extern_with_shape("input_conv2_beta")
            f.write("\n")

            # Encoder convolution weights (64->128->128)
            f.write("// Encoder Convolution Block: 64 -> 128 -> 128 channels\n")
            write_extern_with_shape("encoder_conv1_weight")
            write_extern_with_shape("encoder_conv1_gamma")
            write_extern_with_shape("encoder_conv1_beta")
            write_extern_with_shape("encoder_conv2_weight")
            write_extern_with_shape("encoder_conv2_gamma")
            write_extern_with_shape("encoder_conv2_beta")
            f.write("\n")

            # Decoder convolution weights (192->64->64)
            f.write("// Decoder Convolution Block: 192 -> 64 -> 64 channels\n")
            write_extern_with_shape("decoder_conv1_weight")
            write_extern_with_shape("decoder_conv1_gamma")
            write_extern_with_shape("decoder_conv1_beta")
            write_extern_with_shape("decoder_conv2_weight")
            write_extern_with_shape("decoder_conv2_gamma")
            write_extern_with_shape("decoder_conv2_beta")
            f.write("\n")

            # Output convolution weights (64->64->64)
            f.write("// Output Convolution Block: 64 -> 64 -> 64 channels\n")
            write_extern_with_shape("output_conv1_weight")
            write_extern_with_shape("output_conv1_gamma")
            write_extern_with_shape("output_conv1_beta")
            write_extern_with_shape("output_conv2_weight")
            write_extern_with_shape("output_conv2_gamma")
            write_extern_with_shape("output_conv2_beta")
            f.write("\n")

            # Final convolution weights (64->5)
            f.write("// Final Convolution: 64 -> 5 channels\n")
            write_extern_with_shape("final_conv_weight")
            write_extern_with_shape("final_conv_bias")
            f.write("\n")

            f.write("#endif // UNET_WEIGHTS_H\n")

        print(f"Generated {output_file}")

def main():
    converter = PyTorchToHLSConverter("./best_modular_blocks")

    # Generate data files first to populate weight_shapes
    converter.generate_hls_weights()
    # Then generate header with actual shapes
    converter.generate_weight_header()

    print("Weight conversion completed!")
    print("Files generated:")
    print("  - unet_weights.h (header with declarations)")
    print("  - unet_weights_data.cpp (weight data)")
    print("\nWeight shapes used:")
    for name, shape in converter.weight_shapes.items():
        print(f"  {name}: {shape}")

if __name__ == "__main__":
    main()

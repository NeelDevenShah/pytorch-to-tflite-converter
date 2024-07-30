import torch
import torchvision
import onnx
import argparse
import tensorflow as tf
from onnx_tf.backend import prepare

import torch.nn as nn
import torchvision.models as models

def main_convertor(model, input_shape, output_path):

    # Create a sample input tensor
    sample_input = torch.randn(*input_shape)  # Batch size 1, 3 channels, 512x512 image

    # Convert PyTorch model to ONNX
    torch.onnx.export(
        model,
        sample_input,
        'model.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        verbose=True
    )

    # Convert ONNX to TensorFlow
    onnx_model = onnx.load('model.onnx')
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph('tf_model')

    # Load the TensorFlow model
    model = tf.saved_model.load('tf_model')

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model('tf_model')
    tflite_model = converter.convert()
    
    save_the_model_to_location(output_path, tflite_model)

def load_model(model_path: str):
    # Currently experiment is performed using the efficientNet, for using this code for anyother model change the below code with the other architecutre
    model = torchvision.models.efficientnet_b4(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4, inplace=False),
        nn.Linear(1792, 5)
    )

    # Load pre-trained weights
    model.load_state_dict(
        torch.load(
            f=model_path,
            map_location=torch.device("cpu"),  # load to CPU
        )
    )
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TensorFlow Lite")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch model file (.pth)")
    parser.add_argument("--input_shape", type=int, nargs=4, default=[1, 3, 512, 512], 
                        help="Input shape for the model (batch_size, channels, height, width)")
    parser.add_argument("--output_path", type=str, default="output", help="Output directory for converted models")
    
    return parser.parse_args()

def save_the_model_to_location(save_path: str, tflite_model):

    # Save the TFLite model
    with open(save_path+'model.tflite', 'wb') as f:
        f.write(tflite_model)
        
def main():
    args = parse_args()
    model = load_model(model_path=args.model_path)
    main_convertor(model, tuple(args.input_shape), args.output_path)

if __name__ == "__main__":
    main()

# PyTorch to TensorFlow Lite Converter

This project provides a tool for converting PyTorch models (.pth) to TensorFlow Lite format. It's particularly designed for EfficientNet models but can be adapted for other architectures.

## Features

- Converts PyTorch models to ONNX format
- Converts ONNX models to TensorFlow format
- Converts TensorFlow models to TensorFlow Lite format
- Supports command-line arguments for flexible usage
- Designed for EfficientNet-B4 models with custom classifiers

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or later
- PyTorch
- TensorFlow
- ONNX
- onnx-tf

## Installation

1. Clone this repository:

```shell
   git clone https://github.com/neeldevenshah/pytorch-to-tflite-converter.git
   cd pytorch-to-tflite-converter
```

2. Install the required packages:
   ```shell
   pip install -r requirements.txt
   ```

## Usage

To use the converter, run the script with the following command-line arguments:
python pth_to_tflite.py --model_path /path/to/your/model.pth --input_shape 1 3 512 512 --output_path /path/to/output/directory

```shell
Arguments:

- `--model_path`: Path to the PyTorch model file (.pth) [Required]
- `--input_shape`: Input shape for the model (batch_size, channels, height, width) [Default: 1 3 512 512]
- `--output_path`: Output directory for the converted model [Default: "output"]
```

Example:

```shell
python pth_to_tflite.py --model_path /home/user/models/EfficientNetB3.pth --input_shape 1 3 512 512 --output_path /home/user/converted_models/
```

## How It Works

1. The script loads the PyTorch model from the specified .pth file.
2. It creates a sample input tensor based on the provided input shape.
3. The PyTorch model is exported to ONNX format.
4. The ONNX model is then converted to TensorFlow format.
5. Finally, the TensorFlow model is converted to TensorFlow Lite format.
6. The resulting TFLite model is saved to the specified output path.

## Customization

The current implementation is designed for EfficientNet-B4 models with a custom classifier. To use it with other model architectures:

1. Modify the `load_model` function in the script to initialize and load your specific model architecture.
2. Adjust the `main_convertor` function if necessary to accommodate any specific requirements of your model during the conversion process.

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are correctly installed.
2. Check that your .pth file is compatible with the model architecture defined in the `load_model` function.
3. Verify that the input shape matches your model's expected input.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the original model format
- TensorFlow team for the TFLite format
- ONNX project for providing the intermediate conversion step

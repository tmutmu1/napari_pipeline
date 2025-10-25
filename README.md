# Napari Pipeline Plugin

A custom Napari plugin for CellProfiler pipeline integration, providing seamless workflow capabilities for image analysis.

## Features

- **CellProfiler Integration**: Direct integration with CellProfiler pipelines
- **Image Processing**: Support for various image formats and processing workflows
- **Measurement Tools**: Comprehensive measurement and analysis capabilities
- **Object Detection**: Primary and secondary object identification
- **Feature Enhancement**: Dynamic feature enhancement and suppression

## Requirements

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

### Prerequisites

First, install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Quick Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Run the installation script:

```bash
./install.sh
```

This will:
- Create a virtual environment using `uv`
- Install all dependencies including Napari
- Install the plugin in editable mode
- Verify the installation

### Manual Installation

If you prefer manual installation:

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate

# Install in editable mode
uv pip install -e .
```

## Usage

### Launch Napari with Plugin

Use the provided run script:

```bash
./run.sh
```

Or manually:

```bash
source .venv/bin/activate
napari
```

### Accessing the Plugin

Once Napari is running:

1. Go to the **Plugins** menu
2. Look for **"CellProfiler Pipeline"**
3. Click to open the plugin widget

## Project Structure

```
napari_pipeline/
├── src/
│   └── napari_pipeline/          # Main plugin package
│       ├── __init__.py
│       ├── _widget.py            # Main widget implementation
│       ├── napari.yaml           # Plugin manifest
│       ├── config.json           # Configuration file
│       ├── modules/              # Plugin modules
│       │   ├── DynamicEnhanceOrSuppressFeatures.py
│       │   ├── DynamicIdentifyPrimaryObjects.py
│       │   ├── IdentifySecondaryObjects.py
│       │   └── Threshold.py
│       ├── CellProfilerImageWrapper.py
│       ├── CellProfilerMeasurementWrapper.py
│       ├── CellProfilerObjectsWrapper.py
│       ├── CellProfilerWorkspaceWrapper.py
│       ├── HDF5FileWrapper.py
│       ├── slicer.py
│       └── utils.py
├── pyproject.toml                # Project configuration
├── install.sh                    # Installation script
├── run.sh                        # Launch script
└── README.md                     # This file
```

## Dependencies

The plugin requires the following packages:

- `napari` - Main Napari framework
- `numpy` - Numerical computing
- `qtpy` - Qt bindings
- `centrosome==1.2.1` - CellProfiler utilities
- `scipy` - Scientific computing
- `h5py` - HDF5 file support
- `scikit-image` - Image processing
- `Pillow` - Image I/O

## Development

### Editable Installation

The plugin is installed in editable mode, so changes to the source code will be reflected immediately when you restart Napari.

### Testing

To test the installation:

```bash
source .venv/bin/activate
python -c "import napari_pipeline; print('Plugin loaded successfully')"
```

## Troubleshooting

### Common Issues

1. **Plugin not appearing in Napari**:
   - Ensure the virtual environment is activated
   - Check that the installation completed successfully
   - Restart Napari after installation

2. **Import errors**:
   - Verify all dependencies are installed: `uv pip list`
   - Reinstall if needed: `uv pip install -e .`

3. **uv not found**:
   - Install uv following the official instructions
   - Ensure uv is in your PATH

### Getting Help

If you encounter issues:

1. Check that all dependencies are properly installed
2. Verify the virtual environment is activated
3. Check the Napari console for error messages
4. Ensure you're using a compatible Python version (3.8+)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

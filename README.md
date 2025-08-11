# ComfyUI Workflow Generator

A Python package for generating ComfyUI workflow APIs from `object_info.json` using AST code generator.
Like ComfyScript, but much more simple. 

## Installation

```bash
pip install comfyui-workflow-generator
```

## Usage

### 1. Generate Workflow API

#### From Command Line

```bash
# Generate from local object_info.json (auto-detected)
comfyui-generate object_info.json -o my_workflow_api.py

# Generate from ComfyUI server (auto-detected)
comfyui-generate http://127.0.0.1:8188 -o workflow_api.py

# Generate with default output name
comfyui-generate object_info.json
```

#### From Python

```python
# Import the generated API
from workflow_api import Workflow
from comfyui_workflow_generator import ComfyUIWorkflowExecutor

# Step 1: Upload image with custom name
executor = ComfyUIWorkflowExecutor("http://127.0.0.1:8188")
uploaded_filename = executor.upload_image(
    "my_very_own_image.png", 
    image_name="my_very_own_image.png"
)

# Step 2: Create workflow
wf = Workflow()

# Load checkpoint
model, clip, _ = wf.CheckpointLoaderSimple(ckpt_name="Illustrious-XL-v1.0.safetensors")

# Load vae, i usually do it via separate node
vae = wf.VAELoader(vae_name="sdxl_vae.safetensors")

# Load uploaded image (use the uploaded filename)
image, mask = wf.LoadImage(image="my_very_own_image.png")

# Encode prompts
positive = wf.CLIPTextEncode(text="1girl, lisa_\(genshin_impact\)", clip=clip)
negative = wf.CLIPTextEncode(text="blurry, low quality", clip=clip)

# Encode image for img2img
latent = wf.VAEEncode(pixels=image, vae=vae)

# Sample
denoized_latent = wf.KSampler(
    model=model,
    seed=42,
    steps=20,
    cfg=8.0,
    sampler_name="euler",
    scheduler="normal",
    positive=positive,
    negative=negative,
    latent_image=latent,
    denoise=0.8
)

# Decode
result_image = wf.VAEDecode(samples=denoized_latent, vae=vae)

# Save
wf.SaveImage(images=result_image, filename_prefix="lisa_from_genshin")

# Step 3: Get workflow JSON (for debugging or manual inspection)
workflow_json = wf.get_workflow()
print("Generated workflow:", workflow_json)

# save it to file, it could be loaded in comfyui
with open("workflow_json.json", "w") as f:
    f.write(workflow_json)

# Step 4: Execute it (do not feed it with json, it expects dict)
results = executor.execute_workflow(wf.workflow_dict, output_dir="./results")
```

## Generated workflow look like this:

![Example workflow in ComfyUI](assets/workflow.png)


## Key Design Principles

### 1. **No Input Manipulation**
The workflow executor executes workflows exactly as provided, without any modifications.

### 1. **No Weird Features**
No nested event loops, no Real Mode, Unreal Mode, or other terms from the jargon of x86 assembler programmers of the 90s. It's just a workflow generator, thicc as brick. 

### 3. **Type Safety**
Generated APIs include proper return type hints:

```python
def CheckpointLoaderSimple(self, ckpt_name: str) ->(MODEL, CLIP, VAE):
    ...

def LoadImage(self, image: str) ->(IMAGE, MASK):
    ...

def KSampler(self, model: MODEL, seed: int, steps: int, cfg: float,
        sampler_name: str, scheduler: str, positive: CONDITIONING, negative:
        CONDITIONING, latent_image: LATENT, denoise: float) ->LATENT:
    ...
```


## Package Structure

```
comfyui_workflow_generator/
├── __init__.py          # Package exports
├── generator.py         # AST-based code generator
├── executor.py          # Workflow execution (no input manipulation)
└── cli.py              # Command-line interface
```

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest

# Format code
black comfyui_workflow_generator/

# Lint code
flake8 comfyui_workflow_generator/
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python run_tests.py --unit        # Unit tests only
python run_tests.py --cli         # CLI tests only  
python run_tests.py --executor    # Executor tests only
python run_tests.py --integration # Integration tests only

# Run with coverage
python -m pytest tests/ --cov=comfyui_workflow_generator --cov-report=html
```

### Test Categories

- **Unit Tests** (`tests/test_generator.py`): Test individual components like workflow generation, type normalization, and AST manipulation
- **CLI Tests** (`tests/test_cli.py`): Test command-line interface functionality and error handling
- **Executor Tests** (`tests/test_executor.py`): Test workflow execution, file uploads, and server communication
- **Integration Tests** (`tests/test_integration.py`): Test complete workflows from generation to execution


#!/usr/bin/env python3
"""
Script to inspect YOLO model architectures (YOLOv7-YOLOv11)
Shows block-by-block Conv2d layer details and categorizes into Backbone/Neck/Head
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import sys
import os

# Try to import ultralytics for YOLOv8+
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. YOLOv8+ models may not work.")

# Try to import YOLOv7 or YOLOv5 for YOLOv7 (YOLOv7 uses similar structure to YOLOv5)
YOLOV5_AVAILABLE = False
YOLOV7_AVAILABLE = False
yolov7_load = None
yolov5_load = None

# First try YOLOv7 repository
try:
    # Check if yolov7 directory exists in parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    yolov7_path = os.path.join(parent_dir, 'yolov7')
    if os.path.exists(yolov7_path):
        sys.path.insert(0, yolov7_path)
        from models.experimental import attempt_load as yolov7_load
        YOLOV7_AVAILABLE = True
        YOLOV5_AVAILABLE = True  # YOLOv7 loader can work for YOLOv5 too
except Exception as e:
    pass

# If YOLOv7 not available, try YOLOv5
if not YOLOV7_AVAILABLE:
    try:
        yolov5_path = os.path.join(os.path.dirname(__file__), '..', 'yolov5')
        if os.path.exists(yolov5_path):
            sys.path.append(yolov5_path)
            from yolov5.models.experimental import attempt_load as yolov5_load
            YOLOV5_AVAILABLE = True
    except:
        pass

if not YOLOV5_AVAILABLE and not YOLOV7_AVAILABLE:
    print("Warning: YOLOv5/YOLOv7 not available. YOLOv7 models may not work.")


def get_all_conv2d_layers(model: nn.Module) -> List[nn.Conv2d]:
    """Get all Conv2d layers in the model."""
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def get_block_conv_layers(block: nn.Module) -> List[nn.Conv2d]:
    """Get all Conv2d layers within a specific block."""
    return [m for m in block.modules() if isinstance(m, nn.Conv2d)]


def group_yolov7_blocks(detection_model: nn.Sequential) -> nn.Sequential:
    """
    Group YOLOv7's individual layers into logical blocks (~23 blocks).
    Strategy: Only group consecutive Convs that end with Concat (E-ELAN pattern).
    Single Convs remain as separate blocks.
    
    Args:
        detection_model: The flat Sequential model from YOLOv7 (106 layers)
        
    Returns:
        A new Sequential with grouped logical blocks (~23 blocks)
    """
    import torch.nn as nn
    
    logical_blocks = []
    i = 0
    
    while i < len(detection_model):
        layer = detection_model[i]
        layer_type = type(layer).__name__
        
        # Skip BatchNorm
        if layer_type == 'BatchNorm2d':
            i += 1
            continue
        
        if layer_type == 'Conv':
            # Check what comes after this Conv (skip BatchNorm)
            j = i + 1
            while j < len(detection_model) and type(detection_model[j]).__name__ == 'BatchNorm2d':
                j += 1
            
            # Check if next is another Conv (potential E-ELAN start)
            if j < len(detection_model) and type(detection_model[j]).__name__ == 'Conv':
                # We have multiple Convs - collect them until we hit Concat or something else
                conv_count = 1
                block_layers = [layer]
                # Add BatchNorms for first Conv
                for k in range(i + 1, j):
                    block_layers.append(detection_model[k])
                
                lookahead = j
                found_concat = False
                
                # Collect consecutive Convs
                while lookahead < len(detection_model):
                    current_layer = detection_model[lookahead]
                    current_type = type(current_layer).__name__
                    
                    if current_type == 'BatchNorm2d':
                        block_layers.append(current_layer)
                        lookahead += 1
                    elif current_type == 'Conv':
                        conv_count += 1
                        block_layers.append(current_layer)
                        lookahead += 1
                        # Add BatchNorm if present
                        while lookahead < len(detection_model) and type(detection_model[lookahead]).__name__ == 'BatchNorm2d':
                            block_layers.append(detection_model[lookahead])
                            lookahead += 1
                    elif current_type == 'Concat':
                        # Found Concat - if we have multiple Convs, it's E-ELAN
                        if conv_count > 1:
                            block_layers.append(current_layer)
                            logical_blocks.append(nn.Sequential(*block_layers))
                            i = lookahead + 1
                            found_concat = True
                            break
                        else:
                            # Single Conv before Concat - shouldn't happen, but treat as single
                            logical_blocks.append(layer)
                            i += 1
                            found_concat = True
                            break
                    else:
                        # Hit something else - if we had multiple Convs, group them
                        # Otherwise, single Conv block
                        if conv_count > 1:
                            logical_blocks.append(nn.Sequential(*block_layers))
                        else:
                            logical_blocks.append(layer)
                        i = lookahead
                        found_concat = True
                        break
                
                if found_concat:
                    continue
                else:
                    # Reached end
                    if conv_count > 1:
                        logical_blocks.append(nn.Sequential(*block_layers))
                    else:
                        logical_blocks.append(layer)
                    break
            
            # Check for MPConv (Conv + MaxPool)
            elif j < len(detection_model) and type(detection_model[j]).__name__ == 'MaxPool2d':
                block_layers = [layer]
                for k in range(i + 1, j + 1):
                    block_layers.append(detection_model[k])
                logical_blocks.append(nn.Sequential(*block_layers))
                i = j + 1
                continue
            else:
                # Single Conv block
                logical_blocks.append(layer)
                i += 1
                continue
        
        # Other layer types
        if layer_type in ['Upsample', 'Concat', 'MaxPool2d'] or 'SPP' in layer_type:
            logical_blocks.append(layer)
            i += 1
            continue
        
        # Default
        logical_blocks.append(layer)
        i += 1
    
    return nn.Sequential(*logical_blocks)


def identify_block_type(block: nn.Module) -> str:
    """
    Identify the type of block: Conv, C2f, Concat, RepVGG, RepConv, ADown, AConv, etc.
    
    Args:
        block: The block module to identify
        
    Returns:
        Block type as string with specific names for unique blocks
    """
    block_type_name = block.__class__.__name__
    block_type_lower = block_type_name.lower()
    
    # Check for C2f block (YOLOv8, YOLOv10)
    if block_type_lower == 'c2f':
        return 'C2f'
    
    # Check for C2f-related blocks
    if 'c2f' in block_type_lower:
        return 'C2f'
    
    # Check for RepNCSPELAN blocks (YOLOv9)
    if 'repncspelan' in block_type_lower:
        return 'RepNCSPELAN'
    
    # Check for RepVGG blocks (YOLOv9)
    if 'repvgg' in block_type_lower or 'rep_vgg' in block_type_lower:
        return 'RepVGG'
    
    # Check for RepConv blocks (YOLOv9)
    if 'repconv' in block_type_lower or 'rep_conv' in block_type_lower:
        return 'RepConv'
    
    # Check for ADown blocks (YOLOv9, YOLOv11)
    if 'adown' in block_type_lower or 'a_down' in block_type_lower:
        return 'ADown'
    
    # Check for AConv blocks (YOLOv9, YOLOv11)
    if 'aconv' in block_type_lower or 'a_conv' in block_type_lower:
        return 'AConv'
    
    # Check for SPPELAN blocks (YOLOv9)
    if 'sppelan' in block_type_lower or 'spp_elan' in block_type_lower:
        return 'SPPELAN'
    
    # Check for ELAN blocks (YOLOv9)
    if 'elan' in block_type_lower and 'spp' not in block_type_lower:
        return 'ELAN'
    
    # Check for C3k2 blocks (YOLOv11)
    if 'c3k2' in block_type_lower:
        return 'C3k2'
    
    # Check for C2PSA blocks (YOLOv11)
    if 'c2psa' in block_type_lower:
        return 'C2PSA'
    
    # Check for Identity blocks
    if 'identity' in block_type_lower:
        return 'Identity'
    
    # Check for Concat block
    if 'concat' in block_type_lower or 'cat' in block_type_lower:
        return 'Concat'
    
    # Check for other common block types
    # SPPF, SPP, Focus, etc. are usually Conv blocks
    if block_type_lower in ['sppf', 'spp', 'focus', 'conv', 'conv2d']:
        return 'Conv'
    
    # Check for specific YOLOv11 blocks
    if 'c2' in block_type_lower and 'c2f' not in block_type_lower:
        # Could be C2, C2x, etc.
        return block_type_name  # Return the actual name
    
    # Check if it has Conv layers but isn't C2f or Concat
    block_convs = get_block_conv_layers(block)
    if len(block_convs) > 0:
        # Check for specific patterns in the block structure
        # Check module names within the block
        module_names = [type(m).__name__.lower() for m in block.modules()]
        
        # If it has RepVGG-like structure
        if any('rep' in name for name in module_names):
            return 'RepVGG'
        
        # If it has ADown-like structure
        if any('adown' in name or 'a_down' in name for name in module_names):
            return 'ADown'
        
        # If it has AConv-like structure
        if any('aconv' in name or 'a_conv' in name for name in module_names):
            return 'AConv'
        
        # It's a regular Conv block
        return 'Conv'
    
    # Check for other unique block types
    # Detect by checking for specific module types
    for module in block.modules():
        module_type = type(module).__name__.lower()
        if 'upsample' in module_type or 'upsampling' in module_type:
            return 'Upsample'
        if 'downsample' in module_type or 'downsampling' in module_type:
            return 'Downsample'
        if 'pool' in module_type:
            return 'Pool'
        if 'detect' in module_type:
            return 'Detect'
    
    # If we can't identify but have a specific class name, return it
    if block_type_name not in ['Module', 'Sequential', 'ModuleList']:
        return block_type_name  # Return the actual class name for unique blocks
    
    # Default to Other if we can't identify
    return 'Other'


def categorize_blocks_yolov8(num_blocks: int) -> Dict[str, List[int]]:
    """Categorize blocks for YOLOv8 architecture."""
    # YOLOv8 typically has:
    # Backbone: blocks 0-9
    # Neck: blocks 10-17
    # Head: blocks 18+
    if num_blocks <= 10:
        return {
            'backbone': list(range(min(10, num_blocks))),
            'neck': [],
            'head': []
        }
    elif num_blocks <= 18:
        return {
            'backbone': list(range(10)),
            'neck': list(range(10, min(18, num_blocks))),
            'head': []
        }
    else:
        return {
            'backbone': list(range(10)),
            'neck': list(range(10, 18)),
            'head': list(range(18, num_blocks))
        }


def categorize_blocks_yolov7(num_blocks: int) -> Dict[str, List[int]]:
    """Categorize blocks for YOLOv7 architecture."""
    # YOLOv7 structure is similar to YOLOv5
    # Typically: Backbone (0-~60%), Neck (~60-~85%), Head (~85-100%)
    backbone_end = int(num_blocks * 0.6)
    neck_end = int(num_blocks * 0.85)
    return {
        'backbone': list(range(backbone_end)),
        'neck': list(range(backbone_end, neck_end)),
        'head': list(range(neck_end, num_blocks))
    }


def categorize_blocks_yolov9(num_blocks: int) -> Dict[str, List[int]]:
    """Categorize blocks for YOLOv9 architecture."""
    # YOLOv9 uses similar structure to YOLOv8
    if num_blocks <= 10:
        return {
            'backbone': list(range(min(10, num_blocks))),
            'neck': [],
            'head': []
        }
    elif num_blocks <= 18:
        return {
            'backbone': list(range(10)),
            'neck': list(range(10, min(18, num_blocks))),
            'head': []
        }
    else:
        return {
            'backbone': list(range(10)),
            'neck': list(range(10, 18)),
            'head': list(range(18, num_blocks))
        }


def categorize_blocks_yolov10(num_blocks: int) -> Dict[str, List[int]]:
    """Categorize blocks for YOLOv10 architecture."""
    # YOLOv10 uses similar structure to YOLOv8
    if num_blocks <= 10:
        return {
            'backbone': list(range(min(10, num_blocks))),
            'neck': [],
            'head': []
        }
    elif num_blocks <= 18:
        return {
            'backbone': list(range(10)),
            'neck': list(range(10, min(18, num_blocks))),
            'head': []
        }
    else:
        return {
            'backbone': list(range(10)),
            'neck': list(range(10, 18)),
            'head': list(range(18, num_blocks))
        }


def categorize_blocks_yolov11(num_blocks: int) -> Dict[str, List[int]]:
    """Categorize blocks for YOLOv11 architecture."""
    # YOLOv11 uses similar structure to YOLOv8
    if num_blocks <= 10:
        return {
            'backbone': list(range(min(10, num_blocks))),
            'neck': [],
            'head': []
        }
    elif num_blocks <= 18:
        return {
            'backbone': list(range(10)),
            'neck': list(range(10, min(18, num_blocks))),
            'head': []
        }
    else:
        return {
            'backbone': list(range(10)),
            'neck': list(range(10, 18)),
            'head': list(range(18, num_blocks))
        }


def inspect_yolo_model(model_path: str, model_name: str = "yolov8s.pt"):
    """
    Inspect a YOLO model and print its architecture.
    
    Args:
        model_path: Path to model file or model name (e.g., 'yolov8s.pt', 'yolov9e.pt')
        model_name: Display name for the model
    """
    print("=" * 80)
    print(f"INSPECTING {model_name.upper()}")
    print("=" * 80)
    
    # Determine YOLO version from model name
    yolov_version = None
    if 'yolov7' in model_name.lower() or 'yolov7' in model_path.lower():
        yolov_version = 7
    elif 'yolov8' in model_name.lower() or 'yolov8' in model_path.lower():
        yolov_version = 8
    elif 'yolov9' in model_name.lower() or 'yolov9' in model_path.lower():
        yolov_version = 9
    elif 'yolov10' in model_name.lower() or 'yolov10' in model_path.lower():
        yolov_version = 10
    elif 'yolov11' in model_name.lower() or 'yolov11' in model_path.lower():
        yolov_version = 11
    
    # Load model with multiple fallback strategies
    detection_model = None
    model_loaded = False
    
    # Strategy 1: Try ultralytics (works for YOLOv8, YOLOv9, YOLOv10, and some YOLOv7/YOLOv11)
    if ULTRALYTICS_AVAILABLE:
        try:
            # For YOLOv11, try different model names if the standard one doesn't work
            model_paths_to_try = [model_path]
            if yolov_version == 11:
                # YOLOv11 might use alternative naming (yolo11 instead of yolov11)
                base_name = model_path.replace('.pt', '')
                model_paths_to_try = [
                    model_path,
                    base_name.replace('yolov11', 'yolo11') + '.pt',  # Alternative naming
                ]
            elif yolov_version == 7:
                # YOLOv7 might also need alternative approaches
                # Try ultralytics first, but have fallbacks ready
                pass
            
            for try_path in model_paths_to_try:
                try:
                    model = YOLO(try_path)
                    torch_model = model.model
                    
                    # Try to get detection model - structure varies by version
                    if hasattr(torch_model, 'model'):
                        detection_model = torch_model.model  # YOLOv8+ structure
                    elif hasattr(torch_model, 'yolo'):
                        detection_model = torch_model.yolo  # Alternative structure
                    elif isinstance(torch_model, nn.Sequential):
                        detection_model = torch_model  # Direct Sequential
                    else:
                        # Try to find Sequential in model
                        for attr_name in ['model', 'yolo', 'backbone', 'neck', 'head']:
                            if hasattr(torch_model, attr_name):
                                attr = getattr(torch_model, attr_name)
                                if isinstance(attr, nn.Sequential) or hasattr(attr, '__len__'):
                                    detection_model = attr
                                    break
                        
                        if detection_model is None:
                            # Last resort: try to access model directly
                            detection_model = torch_model
                    
                    if detection_model is not None:
                        model_loaded = True
                        if try_path != model_path:
                            print(f"ℹ️  Loaded model using alternative path: {try_path}")
                        break
                except Exception as e:
                    if try_path == model_paths_to_try[-1]:
                        # Last attempt failed, will try other methods
                        ultralytics_error = e
                    continue
                    
        except Exception as ultralytics_error:
            # Will try fallback methods below
            pass
    
    # Strategy 2: For YOLOv7, try multiple approaches
    if not model_loaded and yolov_version == 7:
        # First, try to download YOLOv7 if file doesn't exist (ultralytics might support it)
        if not os.path.exists(model_path) and ULTRALYTICS_AVAILABLE:
            try:
                print("ℹ️  YOLOv7 file not found locally, trying to download via ultralytics...")
                # Try with ultralytics - it might download YOLOv7
                model = YOLO(model_path)
                torch_model = model.model
                if hasattr(torch_model, 'model'):
                    detection_model = torch_model.model
                else:
                    detection_model = torch_model
                model_loaded = True
                print("✅ Successfully loaded YOLOv7 via ultralytics")
            except Exception as download_error:
                print(f"⚠️  Ultralytics download failed: {download_error}")
        
        # Try YOLOv7 or YOLOv5 loader if available
        if not model_loaded and (YOLOV7_AVAILABLE or YOLOV5_AVAILABLE):
            try:
                if YOLOV7_AVAILABLE:
                    print("ℹ️  Trying YOLOv7 loader...")
                    model = yolov7_load(model_path, device='cpu')
                else:
                    print("ℹ️  Trying YOLOv5 loader for YOLOv7...")
                    model = yolov5_load(model_path, device='cpu')
                
                if hasattr(model, 'model'):
                    detection_model = model.model
                else:
                    detection_model = model
                model_loaded = True
                loader_name = "YOLOv7" if YOLOV7_AVAILABLE else "YOLOv5"
                print(f"✅ Successfully loaded YOLOv7 via {loader_name} loader")
            except Exception as loader_error:
                print(f"⚠️  Loader failed: {loader_error}")
        
        # If YOLOv5 loader failed, try direct torch.load (YOLOv7 models are often saved as state dicts)
        if not model_loaded:
            # Try alternative file names
            alternative_paths = [
                model_path,
                model_path.replace('-', '_'),  # yolov7-tiny.pt -> yolov7_tiny.pt
                model_path.replace('_', '-'),  # yolov7_tiny.pt -> yolov7-tiny.pt
            ]
            
            for try_path in alternative_paths:
                if os.path.exists(try_path):
                    try:
                        print(f"ℹ️  Trying to load YOLOv7 from {try_path}...")
                        # YOLOv7 models are pickled and require the 'models' module
                        # We'll create a proper dummy models module structure to allow loading
                        import sys
                        import types
                        
                        # Create a proper dummy 'models' module structure
                        if 'models' not in sys.modules:
                            dummy_models = types.ModuleType('models')
                            sys.modules['models'] = dummy_models
                            
                            # Create models.yolo submodule
                            dummy_yolo = types.ModuleType('models.yolo')
                            sys.modules['models.yolo'] = dummy_yolo
                            
                            # Add common YOLOv7 model classes as dummy classes
                            class DummyModel(nn.Module):
                                def __init__(self, *args, **kwargs):
                                    super().__init__()
                            
                            class DummyYOLO(nn.Module):
                                def __init__(self, *args, **kwargs):
                                    super().__init__()
                            
                            dummy_models.Model = DummyModel
                            dummy_models.Ensemble = DummyModel
                            dummy_models.attempt_load = lambda *args, **kwargs: None
                            dummy_yolo.Model = DummyYOLO
                        
                        try:
                            checkpoint = torch.load(try_path, map_location='cpu', weights_only=False)
                            
                            # Extract model from checkpoint
                            if isinstance(checkpoint, dict):
                                if 'model' in checkpoint:
                                    model_obj = checkpoint['model']
                                    if hasattr(model_obj, 'model'):
                                        detection_model = model_obj.model
                                    elif isinstance(model_obj, nn.Module):
                                        # Try to find Sequential in the model
                                        for attr_name in ['model', 'yolo', 'backbone']:
                                            if hasattr(model_obj, attr_name):
                                                attr = getattr(model_obj, attr_name)
                                                if isinstance(attr, (nn.Sequential, nn.ModuleList)) or hasattr(attr, '__len__'):
                                                    detection_model = attr
                                                    break
                                        if detection_model is None:
                                            detection_model = model_obj
                            elif isinstance(checkpoint, nn.Module):
                                # Direct model object
                                if hasattr(checkpoint, 'model'):
                                    detection_model = checkpoint.model
                                else:
                                    detection_model = checkpoint
                            
                            if detection_model is not None:
                                model_loaded = True
                                print(f"✅ Successfully loaded YOLOv7 from {try_path}")
                                break
                        except Exception as load_error:
                            # If loading fails, try to extract info from state_dict keys
                            if 'No module named' in str(load_error) or 'models' in str(load_error):
                                print(f"⚠️  YOLOv7 checkpoint requires the YOLOv7 repository.")
                                print(f"   Attempting to extract architecture from state_dict...")
                                try:
                                    # Try to extract just the weights by reading the file differently
                                    # This is a workaround - we'll analyze what we can
                                    print(f"   Note: Full inspection requires YOLOv7 repo: https://github.com/WongKinYiu/yolov7")
                                    print(f"   For now, you can use YOLOv5 loader if you have YOLOv5 repository available.")
                                except:
                                    pass
                            print(f"⚠️  Failed to load {try_path}: {load_error}")
                            continue
                    except Exception as general_error:
                        print(f"⚠️  Error processing {try_path}: {general_error}")
                        continue
    
    # Strategy 3: For YOLOv11, if ultralytics failed, inform user
    if not model_loaded and yolov_version == 11:
        print(f"⚠️  YOLOv11 may not be fully supported by ultralytics yet.")
        print(f"   Trying alternative approaches...")
        # Try to load as if it's a YOLOv8 model structure
        try:
            if os.path.exists(model_path):
                # Try direct torch.load
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model_obj = checkpoint['model']
                    if hasattr(model_obj, 'model'):
                        detection_model = model_obj.model
                        model_loaded = True
        except Exception as e:
            print(f"⚠️  Alternative loading failed: {e}")
    
    # Final check
    if not model_loaded or detection_model is None:
        print(f"❌ Could not load model using any available method")
        print(f"   Tried: ultralytics, YOLOv5 loader (for YOLOv7), direct torch.load")
        if yolov_version == 11:
            print(f"   Note: YOLOv11 may not be fully supported by ultralytics yet.")
            print(f"   You may need to wait for ultralytics to add YOLOv11 support,")
            print(f"   or use a YOLOv8/YOLOv10 model instead.")
        elif yolov_version == 7:
            print(f"\n   For YOLOv7, you need one of the following:")
            print(f"   1. YOLOv7 repository cloned and in Python path:")
            print(f"      git clone https://github.com/WongKinYiu/yolov7.git")
            print(f"      export PYTHONPATH=$PYTHONPATH:$(pwd)/yolov7")
            print(f"   2. YOLOv5 repository (YOLOv7 is based on YOLOv5):")
            print(f"      git clone https://github.com/ultralytics/yolov5.git")
            print(f"      The script will automatically detect it if in the parent directory")
            print(f"\n   Once the repository is available, run the script again.")
        return
    
    if detection_model is None:
        print("❌ Could not access detection model")
        return
    
    # For YOLOv7, show the raw layer structure without grouping
    # YOLOv7 has a flat Sequential structure where each layer is separate
    # This is different from YOLOv8+ which groups operations into logical blocks
    if yolov_version == 7 and isinstance(detection_model, nn.Sequential):
        print(f"ℹ️  YOLOv7 architecture: {len(detection_model)} individual layers\n")
        print("   Note: YOLOv7 uses a flat Sequential structure where each layer")
        print("   (Conv, Concat, Pool, Upsample, etc.) is stored as a separate entry.")
        print("   This is different from YOLOv8+ which groups related operations into logical blocks.\n")
        if False:  # Disable grouping - show raw structure
            print(f"ℹ️  Grouped YOLOv7 layers: {len(original_detection_model)} layers → {len(detection_model)} logical blocks")
    
    # Get all blocks - handle different model structures
    num_blocks = None
    if isinstance(detection_model, nn.Sequential):
        num_blocks = len(detection_model)
    elif hasattr(detection_model, '__len__'):
        try:
            num_blocks = len(detection_model)
        except:
            pass
    
    # If we still don't have num_blocks, try to find blocks in the model
    if num_blocks is None:
        # Try to find a Sequential or ModuleList in the model
        for module in detection_model.modules() if hasattr(detection_model, 'modules') else []:
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                num_blocks = len(module)
                detection_model = module
                break
        
        # If still not found, try to access model attributes
        if num_blocks is None:
            # For YOLOv7, the structure might be different
            if hasattr(detection_model, 'model'):
                detection_model = detection_model.model
                if isinstance(detection_model, nn.Sequential):
                    num_blocks = len(detection_model)
                elif hasattr(detection_model, '__len__'):
                    num_blocks = len(detection_model)
    
    if num_blocks is None:
        print("❌ Could not determine number of blocks")
        print(f"   Model type: {type(detection_model)}")
        print(f"   Model attributes: {dir(detection_model)[:10]}")
        return
    
    # Get all Conv2d layers
    all_conv_layers = get_all_conv2d_layers(detection_model)
    total_conv_layers = len(all_conv_layers)
    
    print(f"\nTotal Conv2d layers in model: {total_conv_layers}")
    print(f"Total blocks in model: {num_blocks}")
    
    # Note about YOLOv7 architecture difference
    if yolov_version == 7:
        print("\n" + "=" * 80)
        print("NOTE: YOLOv7 Architecture Structure")
        print("=" * 80)
        print("YOLOv7 uses a modular architecture where each individual layer")
        print("(Conv, Concat, Pool, Upsample, etc.) is stored as a separate entry")
        print("in the Sequential model, resulting in ~106 individual 'blocks'.")
        print("\nLogically, these can be grouped into ~23 blocks:")
        print("  - E-ELAN blocks: Multiple Conv layers ending with Concat")
        print("  - MPConv blocks: Conv + MaxPool combinations")
        print("  - Single Conv layers")
        print("  - Upsample/Concat/Pool as separate blocks")
        print("\nThe architecture shown above reflects the actual model structure.")
        print("For a grouped view, you would need to manually group layers 2-10,")
        print("11-16, 17-23, etc. into logical E-ELAN blocks.\n")
    else:
        print()
    
    # Categorize blocks
    if yolov_version == 7:
        categories = categorize_blocks_yolov7(num_blocks)
    elif yolov_version == 8:
        categories = categorize_blocks_yolov8(num_blocks)
    elif yolov_version == 9:
        categories = categorize_blocks_yolov9(num_blocks)
    elif yolov_version == 10:
        categories = categorize_blocks_yolov10(num_blocks)
    elif yolov_version == 11:
        categories = categorize_blocks_yolov11(num_blocks)
    else:
        # Default to YOLOv8 structure
        categories = categorize_blocks_yolov8(num_blocks)
    
    # Analyze each block
    block_info = []
    for block_idx in range(num_blocks):
        if isinstance(detection_model, nn.Sequential):
            block = detection_model[block_idx]
        elif hasattr(detection_model, '__getitem__'):
            block = detection_model[block_idx]
        else:
            continue
        
        # Identify block type
        block_type = identify_block_type(block)
        
        block_convs = get_block_conv_layers(block)
        num_convs = len(block_convs)
        
        # Build layer details string
        layer_details = []
        total_channels = 0
        for conv_idx, conv in enumerate(block_convs):
            out_channels = conv.weight.shape[0]
            layer_details.append(f"Conv{conv_idx}({out_channels})")
            total_channels += out_channels
        
        layer_details_str = ", ".join(layer_details) if layer_details else "No Conv2d layers"
        
        block_info.append({
            'block_idx': block_idx,
            'block_type': block_type,
            'num_convs': num_convs,
            'layer_details': layer_details_str,
            'total_channels': total_channels
        })
    
    # Print table - match the exact format from user's example, with block type added
    header = "Block  Type     Conv2d Layers   Layer Details                                      Total Channels "
    print(header)
    print("-" * len(header))
    
    for info in block_info:
        block_idx = info['block_idx']
        block_type = info['block_type']
        num_convs = info['num_convs']
        layer_details = info['layer_details']
        total_channels = info['total_channels']
        
        # Show full layer details - don't truncate
        # The format allows layer details to extend beyond the header width if needed
        # Block (6 chars), Type (10 chars), Conv2d Layers (16 chars), Layer Details (flexible), Total Channels (15 chars)
        print(f"{block_idx:<6} {block_type:<10} {num_convs:<16} {layer_details:<50} {total_channels:<15}")
    
    print("-" * len(header))
    print(f"TOTAL  {'':<10} {total_conv_layers:<16}")
    
    # Print categorization
    print("\n" + "=" * 80)
    print("BLOCK CATEGORIZATION:")
    print("=" * 80)
    
    backbone_blocks = categories['backbone']
    neck_blocks = categories['neck']
    head_blocks = categories['head']
    
    if backbone_blocks:
        backbone_range = f"{backbone_blocks[0]}-{backbone_blocks[-1]}" if len(backbone_blocks) > 1 else str(backbone_blocks[0])
        print(f"\nBackbone blocks ({backbone_range}): {backbone_blocks}")
    if neck_blocks:
        neck_range = f"{neck_blocks[0]}-{neck_blocks[-1]}" if len(neck_blocks) > 1 else str(neck_blocks[0])
        print(f"Neck blocks ({neck_range}): {neck_blocks}")
    if head_blocks:
        head_range = f"{head_blocks[0]}+" if len(head_blocks) > 0 else ""
        print(f"Head blocks ({head_range}): {head_blocks}")
    
    # Count Conv2d layers per category
    backbone_convs = sum(block_info[i]['num_convs'] for i in backbone_blocks if i < len(block_info))
    neck_convs = sum(block_info[i]['num_convs'] for i in neck_blocks if i < len(block_info))
    head_convs = sum(block_info[i]['num_convs'] for i in head_blocks if i < len(block_info))
    
    print("\nSUMMARY:")
    print(f"Backbone Conv2d layers: {backbone_convs}")
    print(f"Neck Conv2d layers: {neck_convs}")
    print(f"Head Conv2d layers: {head_convs}")
    print(f"Total Conv2d layers: {total_conv_layers}")
    
    # Count block types
    block_type_counts = {}
    block_type_indices = {}
    for info in block_info:
        block_type = info['block_type']
        block_idx = info['block_idx']
        block_type_counts[block_type] = block_type_counts.get(block_type, 0) + 1
        if block_type not in block_type_indices:
            block_type_indices[block_type] = []
        block_type_indices[block_type].append(block_idx)
    
    print("\nBLOCK TYPE SUMMARY:")
    for block_type in sorted(block_type_counts.keys()):
        count = block_type_counts[block_type]
        indices = block_type_indices[block_type]
        print(f"{block_type} blocks: {count} - {indices}")
    print()


def main():
    """Main function to inspect YOLOv7 models."""
    
    # List of YOLOv7 models to inspect
    models_to_inspect = [
        ("yolov7.pt", "YOLOv7"),
        ("yolov7-tiny.pt", "YOLOv7-tiny"),
        # Alternative names that might be used
        ("yolov7_tiny.pt", "YOLOv7-tiny"),
    ]
    
    print("=" * 80)
    print("YOLOv7 ARCHITECTURE INSPECTION")
    print("=" * 80)
    print("\nThis script will attempt to load and inspect YOLOv7 models.")
    print("\n⚠️  IMPORTANT: YOLOv7 models require the YOLOv7 repository to load properly.")
    print("   The models are pickled and need the original model definitions.")
    print("\nTo inspect YOLOv7 models, you have two options:")
    print("  1. Clone YOLOv7 repository:")
    print("     git clone https://github.com/WongKinYiu/yolov7.git")
    print("     Then ensure it's in your Python path")
    print("  2. Use YOLOv5 loader (YOLOv7 is based on YOLOv5):")
    print("     Clone YOLOv5: git clone https://github.com/ultralytics/yolov5.git")
    print("\nThe script will try multiple loading methods if models are found locally.\n")
    
    for model_path, model_name in models_to_inspect:
        try:
            inspect_yolo_model(model_path, model_name)
            print("\n" + "=" * 80 + "\n")
        except Exception as e:
            print(f"❌ Failed to inspect {model_name}: {e}\n")
            print("=" * 80 + "\n")
            continue


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect YOLOv7 model architectures")
    parser.add_argument("--model", type=str, default=None, 
                       help="Specific model to inspect (e.g., 'yolov7.pt', 'yolov7-tiny.pt')")
    parser.add_argument("--all", action="store_true",
                       help="Inspect all YOLOv7 models (yolov7.pt and yolov7-tiny.pt)")
    
    args = parser.parse_args()
    
    if args.model:
        inspect_yolo_model(args.model, args.model)
    else:
        main()


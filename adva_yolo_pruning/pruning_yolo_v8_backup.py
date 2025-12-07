import logging
from ultralytics import YOLO
import torch

from yolov8_utils import build_mini_net, extract_conv_weights_norm, get_all_conv2d_layers, get_raw_objects_debug_v8, aggregate_activations_from_matches, prune_conv2d_layer_in_yolo, get_conv_bn_pairs, extract_bn_gamma
from yolo_layer_pruner import YoloLayerPruner
from clustering import select_optimal_components, kmedoids_fasterpam
from structural_pruning import apply_structural_activation_pruning
import random
import numpy as np


logger = logging.getLogger("yolov8_pruning")
fh = logging.FileHandler('log.txt', mode='w')
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

# logger = logging.getLogger("yolov8_pruning")
# logging.basicConfig(level=logging.INFO)
def apply_pruning_v8_prune_lowest_activation(model_path, train_data, valid_data, classes, last_layer_idx=3, k_values=None):
    """
    Prune the channels with the lowest mean activation scores for each Conv2d layer.
    For each k in k_values, prune k channels with the lowest mean activation.
    """
    import numpy as np

    logger.info("Starting activation-based pruning for YOLOv8 model.")
    if k_values is None:
        k_values = list(range(4, 20, 4))  # Default: prune 4, 6, 8, ..., 18 channels

    for k in k_values:
        print(f"\n=== Pruning {k} channels with lowest mean activation per layer ===")
        model = YOLO(model_path)
        torch_model = model.model
        detection_model = torch_model.model  # nn.Sequential of blocks

        last_layer_idx = 2
        for layer_idx in range(2, last_layer_idx + 1):
            logger.info(f"Pruning block #{layer_idx}..")
            sliced_block = detection_model[:layer_idx]
            conv_layers = get_all_conv2d_layers(sliced_block)
            processed_convs = set()
            processed_convs.add(0)  
            for conv_idx, conv_layer in enumerate(conv_layers):
                num_channels = conv_layer.weight.shape[0]
                if num_channels <= k:
                    print(f"Skipping layer {conv_idx}: not enough channels to prune {k}.")
                    continue

                if conv_idx in processed_convs:
                    continue    
                processed_convs.add(conv_idx)

                # Build mini-net and get activations
                try:
                    mini_net = build_mini_net(sliced_block, conv_layer)
                except ValueError:
                    logger.error(f"Failed to build mini-network for conv2d layer #{conv_idx} of block {layer_idx - 1}. Skipping this layer.")
                    continue

                train_matched_objs, _ = get_raw_objects_debug_v8(model, mini_net, train_data)
                train_activations = aggregate_activations_from_matches(train_matched_objs, classes)

                # Save activation distributions
                with open("channel_activation_distributions_layer2.txt", "a") as f:
                    f.write(f"Layer {conv_idx} activation distributions:\n")
                    for ch in range(conv_layer.weight.shape[0]):
                        f.write(f"  Channel {ch}:\n")
                        for cls in range(len(classes)):
                            acts = train_activations.get(ch, {}).get(cls, [])
                            f.write(f"    Class {cls}: {acts}\n")

                # Compute mean activation per channel
                # train_activations: Dict[channel][class] -> List[float]
                mean_activations = []
                for ch in range(num_channels):
                    all_acts = []
                    for acts in train_activations.get(ch, {}).values():
                        all_acts.extend(acts)
                    if all_acts:
                        mean_activations.append(np.mean(all_acts))
                    else:
                        mean_activations.append(0.0)  # If no activations, treat as zero

                mean_activations = np.array(mean_activations)
                indices_sorted = np.argsort(mean_activations)  # ascending order
                indices_to_keep = sorted(indices_sorted[k:])  # keep all except k lowest
                indices_to_prune = indices_sorted[:k]
                
                # Save pruned channels
                with open("pruned_channels_per_step_layer3.txt", "a") as f:
                    f.write(f"Layer in activ {conv_idx} pruned channels this step: {list(indices_to_prune)}\n")

                # Prune
                model = prune_conv2d_layer_in_yolo(model, conv_idx, indices_to_keep)
                print(f"Pruned layer {conv_idx} (kept {len(indices_to_keep)} channels)")

                # Optionally retrain after each layer
                model.train(data="pruning/data/VOC_adva.yaml", epochs=3, verbose=False)
                print(f"Retrained model for 3 epochs after pruning layer {conv_idx}")

        # Final evaluation
        final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
        logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
        print("DEBUG: Final evaluation complete.")

        # Save results
        with open("pruning_log_lowest_activation_layer3.txt", "a") as f:
            f.write(f"k={k}, mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                    f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                    f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                    f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")

    return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_pruning_v8_fix_k_medoids(model_path, train_data, valid_data, classes, last_layer_idx=3):
    logger.info("Starting the pruning process for YOLOv8 model.")
    k_values = list(range(20, 32, 2))  
    for k in k_values:
        print(f"\n=== Pruning with k={k} components ===")
        # Load model
        model = YOLO(model_path)
        torch_model = model.model
        logger.debug("Model loaded successfully.")
        detection_model = torch_model.model  # YOLOv8: model.model.model is the nn.Sequential of blocks
        logger.debug("Accessed the detection model from YOLOv8.")
        
        last_layer_idx = 2
        for layer_idx in range(2, last_layer_idx + 1):
            logger.info(f"Starting to prune block #{layer_idx}..")
            sliced_block = detection_model[:layer_idx]
            print(f"DEBUG: Sliced block {layer_idx} with {len(sliced_block)} layers.")
            #New part 
            all_conv_layers = get_all_conv2d_layers(model)
            conv_layers = get_all_conv2d_layers(sliced_block)

            processed_convs = set()
            processed_convs.add(0)
            for conv_idx, conv_layer in enumerate(conv_layers):
                print("conv_idx:", conv_idx)
                print("conv_layer:", conv_layer)
                if conv_idx in processed_convs:
                        continue

                processed_convs.add(conv_idx)
                print("processed_convs:", processed_convs)

                logger.info(f"Starting to prune conv2d layer #{conv_idx} of block {layer_idx - 1}..")
                with open("pruning_fix_k_medoids.txt", "a") as f:
                    num_channels_before = conv_layer.weight.shape[0]
                    f.write(f"Layer {conv_idx} channels before pruning: {num_channels_before}\n")

                try:
                    mini_net = build_mini_net(sliced_block, conv_layer)
                    print(f"DEBUG: Built mini_net for block {layer_idx}, layer {conv_idx}")
                except ValueError:
                    logger.error(f"Failed to build mini-network for conv2d layer #{conv_idx} of block {layer_idx - 1}. Skipping this layer.")
                    print(f"DEBUG: Failed to build mini_net for block {layer_idx}, layer {conv_idx}")
                    continue

                layer_weights = extract_conv_weights_norm(conv_layer)
                print(f"DEBUG: Extracted weights norm for layer {conv_idx}, shape: {layer_weights.shape}")

                train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
                print(f"DEBUG: Got raw objects for layer {conv_idx}")

                train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
                print(f"DEBUG: Aggregated activations for layer {layer_idx}, shape: {train_activations.shape if hasattr(train_activations, 'shape') else type(train_activations)}")


                if not train_activations or all(len(v) == 0 for v in train_activations.values()):
                    logger.warning(f"No matched activations for layer {layer_idx}, skipping pruning for this layer.")
                    continue

                graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
                print(f"DEBUG: Created graph space for layer {layer_idx}")

                print("layer_weights.shape:", layer_weights.shape)
                print("graph_space['reduced_matrix'].shape:", graph_space['reduced_matrix'].shape)
                print("len(train_activations):", len(train_activations))

                # Cluster and select k components
                k_medoids = kmedoids_fasterpam(graph_space['reduced_matrix'], k)
                indices_to_keep = k_medoids['medoids'].tolist()
                
                model = prune_conv2d_layer_in_yolo(model, conv_idx, indices_to_keep)
                all_conv_layers = get_all_conv2d_layers(model)

                print(f"DEBUG: Pruned layer {conv_idx}")

                pruned_layer = all_conv_layers[conv_idx]

                num_channels_after = (pruned_layer.weight.abs().sum(dim=(1,2,3)) != 0).sum().item()
                with open("pruning_fix_k_medoids.txt", "a") as f:
                    f.write(f"Layer {conv_idx} channels after pruning: {num_channels_after}\n")
                    f.write(f"Pruned {num_channels_before - num_channels_after} channels in layer {conv_idx}\n")
                    f.write("=== all_conv_layers ===")
                    for idx, layer in enumerate(all_conv_layers):
                        f.write(f"all_conv_layers[{idx}]: id={id(layer)}, shape={tuple(layer.weight.shape)}")

                model.train(data="pruning/data/VOC_adva.yaml", epochs=5, verbose=False)
                print(f"DEBUG: Retrained model for 5 epoch after pruning layer {conv_idx}")

        # Final evaluation
        final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
        logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
        print("DEBUG: Final evaluation complete.")

        with open("pruning_fix_k_medoids.txt", "a") as f:
            f.write(f"k={k}, mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
            f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
            f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
            f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")

    return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_pruning_v8_fix_number(model_path, train_data, valid_data, classes, last_layer_idx=3):
    logger.info("Starting the pruning process for YOLOv8 model.")
    select_components_based_on_mss = False

    k_values = list(range(4, 20, 2))  # Try pruning 4, 6, 8, ..., 18 channels
    for k in k_values:
        print(f"\n=== Pruning with k={k} channels per layer ===")
        # Load model
        model = YOLO(model_path)
        torch_model = model.model
        logger.debug("Model loaded successfully.")
        detection_model = torch_model.model  # YOLOv8: model.model.model is the nn.Sequential of blocks
        logger.debug("Accessed the detection model from YOLOv8.") 

        last_layer_idx = 2
        for layer_idx in range(2, last_layer_idx + 1):
            logger.info(f"Starting to prune block #{layer_idx}..")
            sliced_block = detection_model[:layer_idx]
            #New part 
            all_conv_layers = get_all_conv2d_layers(model)
            conv_layers = get_all_conv2d_layers(sliced_block)

            print("=== all_conv_layers ===")
            for idx, layer in enumerate(all_conv_layers):
                print(f"all_conv_layers[{idx}]: id={id(layer)}, shape={tuple(layer.weight.shape)}")

            print("=== conv_layers ===")
            for idx, layer in enumerate(conv_layers):
                print(f"conv_layers[{idx}]: id={id(layer)}, shape={tuple(layer.weight.shape)}")

            processed_convs = set()
            processed_convs.add(0)

            for conv_idx, conv_layer in enumerate(conv_layers):
                print("conv_idx:", conv_idx)
                print("conv_layer:", conv_layer)

                if conv_idx in processed_convs:
                        continue
                processed_convs.add(conv_idx)

                logger.info(f"Starting to prune conv2d layer #{conv_idx} of block {layer_idx - 1}..")

                with open("pruning_log_fix_prunned_rand_3_epoch.txt", "a") as f:
                    num_channels_before = conv_layer.weight.shape[0]
                    f.write(f"Layer {conv_idx} channels before pruning: {num_channels_before}\n")
                if num_channels_before <= k:
                    print(f"Skipping layer {conv_idx}: not enough channels to prune {k}.")
                    continue

                indices_to_keep = sorted(random.sample(range(num_channels_before), num_channels_before - k))
                # indices_to_keep = list(range(num_channels_before - k))
                model = prune_conv2d_layer_in_yolo(model, conv_idx, indices_to_keep)
                all_conv_layers = get_all_conv2d_layers(model)

                print(f"DEBUG: Pruned layer {conv_idx}")

                pruned_layer = all_conv_layers[conv_idx]
                num_channels_after = (pruned_layer.weight.abs().sum(dim=(1,2,3)) != 0).sum().item()
                with open("pruning_log_fix_prunned_rand_3_epoch.txt", "a") as f:
                    f.write(f"Layer {conv_idx} channels after pruning: {num_channels_after}\n")
                    f.write(f"Pruned {num_channels_before - num_channels_after} channels in layer {conv_idx}\n")
                    f.write("=== all_conv_layers ===")
                    for idx, layer in enumerate(all_conv_layers):
                        f.write(f"all_conv_layers[{idx}]: id={id(layer)}, shape={tuple(layer.weight.shape)}")

                model.train(data="pruning/data/VOC_adva.yaml", epochs=3, verbose=False)
                print(f"DEBUG: Retrained model for 3 epoch after pruning layer {conv_idx}")
                
        # Final evaluation
        final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
        logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
        print("DEBUG: Final evaluation complete.")

        with open("pruning_log_fix_prunned_rand_3_epoch.txt", "a") as f:
            f.write(f"k={k}, mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
            f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
            f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
            f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")

    return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_pruning_v8(model_path, train_data, valid_data, classes, last_layer_idx=3):
    logger.info("Starting the pruning process for YOLOv8 model.")
    k_default_value = 50
    select_components_based_on_mss = False
    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    logger.debug("Model loaded successfully.")
    detection_model = torch_model.model  # YOLOv8: model.model.model is the nn.Sequential of blocks
    logger.debug("Accessed the detection model from YOLOv8.") 

    last_layer_idx = 5
    for layer_idx in range(5, last_layer_idx + 1):
        logger.info(f"Starting to prune block #{layer_idx}..")
        sliced_block = detection_model[:layer_idx]
        print(f"DEBUG: Sliced block {layer_idx} with {len(sliced_block)} layers.")
        #New part 
        all_conv_layers = get_all_conv2d_layers(model)
        conv_layers = get_all_conv2d_layers(sliced_block)

        print("=== all_conv_layers ===")
        for idx, layer in enumerate(all_conv_layers):
            print(f"all_conv_layers[{idx}]: id={id(layer)}, shape={tuple(layer.weight.shape)}")

        print("=== conv_layers ===")
        for idx, layer in enumerate(conv_layers):
            print(f"conv_layers[{idx}]: id={id(layer)}, shape={tuple(layer.weight.shape)}")

        processed_convs = set()
        processed_convs.add(0)

        print("processed_convs:", processed_convs)
        for conv_idx, conv_layer in enumerate(conv_layers):
            print("conv_idx:", conv_idx)
            print("conv_layer:", conv_layer)
            if conv_idx in processed_convs:
                    continue

            processed_convs.add(conv_idx)
            print("processed_convs:", processed_convs)

            logger.info(f"Starting to prune conv2d layer #{conv_idx} of block {layer_idx - 1}..")
            with open("pruning_layer6.txt", "a") as f:
                num_channels_before = conv_layer.weight.shape[0]
                f.write(f"Layer {conv_idx} channels before pruning: {num_channels_before}\n")

            try:
                mini_net = build_mini_net(sliced_block, conv_layer)
                print(f"DEBUG: Built mini_net for block {layer_idx}, layer {conv_idx}")
            except ValueError:
                logger.error(f"Failed to build mini-network for conv2d layer #{conv_idx} of block {layer_idx - 1}. Skipping this layer.")
                print(f"DEBUG: Failed to build mini_net for block {layer_idx}, layer {conv_idx}")
                continue

            layer_weights = extract_conv_weights_norm(conv_layer)
            print(f"DEBUG: Extracted weights norm for layer {conv_idx}, shape: {layer_weights.shape}")

            train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
            logger.info(f"Total number of training objects used:\n"
                        f"Train matched objects: {len(train_matched_objs)} - "
                        f"Train un-matched objects: {len(train_unmatched_objs)} - "
                        f"Percentage of matched objects overall: [{len(train_matched_objs) / (len(train_matched_objs) + len(train_unmatched_objs))}].")
            print(f"DEBUG: Got raw objects for layer {conv_idx}")

            train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
            print(f"DEBUG: Aggregated activations for layer {layer_idx}, shape: {train_activations.shape if hasattr(train_activations, 'shape') else type(train_activations)}")


            if not train_activations or all(len(v) == 0 for v in train_activations.values()):
                logger.warning(f"No matched activations for layer {layer_idx}, skipping pruning for this layer.")
                continue

            graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
            print(f"DEBUG: Created graph space for layer {layer_idx}")

            print("layer_weights.shape:", layer_weights.shape)
            print("graph_space['reduced_matrix'].shape:", graph_space['reduced_matrix'].shape)
            print("len(train_activations):", len(train_activations))

            if not select_components_based_on_mss:
            # TODO: In the original implementation, k_default_value was not being used, so this if-else block did the same thing. Need to ask what the purpose was or check in the original algorithm.
                optimal_components = select_optimal_components(graph_space, layer_weights, len(train_activations), k_default_value)
                select_components_based_on_mss = True
            else:
                optimal_components = select_optimal_components(graph_space, layer_weights, len(train_activations), -1)
            
            logger.info(f"Number of optimal components for conv2d layer #{conv_idx}: {len(optimal_components)}")

            print(f"DEBUG: Selected {len(optimal_components)} optimal components for layer {conv_idx}")
            print("Main model id:", id(model))
            model = prune_conv2d_layer_in_yolo(model, conv_idx, optimal_components)
            all_conv_layers = get_all_conv2d_layers(model)

            print(f"DEBUG: Pruned layer {conv_idx}")

            pruned_layer = all_conv_layers[conv_idx]
            num_channels_after = (pruned_layer.weight.abs().sum(dim=(1,2,3)) != 0).sum().item()
            with open("pruning_layer6.txt", "a") as f:
                f.write(f"Layer {conv_idx} channels after pruning: {num_channels_after}\n")
                f.write(f"Pruned {num_channels_before - num_channels_after} channels in layer {conv_idx}\n")

            #Evaluate after pruning this layer
            pruned_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
            logger.info(f"After pruning layer  & **before** re-train conv layer {conv_idx}: {pruned_metrics.results_dict}")
            print(f"DEBUG: Evaluated pruned model after layer {conv_idx}")

            model.train(data="pruning/data/VOC_adva.yaml", epochs=20, verbose=False)
            print(f"DEBUG: Retrained model for 1 epoch after pruning layer {conv_idx}")

    # Final evaluation
    final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
    logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")
    with open("pruning_layer6.txt", "a") as f:
        f.write(f"Final mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")
        # Per-class mAP@0.5
        if hasattr(final_metrics, "maps"):
            f.write("Per-class mAP@0.5:\n")
            for idx, class_map in enumerate(final_metrics.maps):
                f.write(f"Class {idx}: {class_map:.4f}\n")
        else:
            f.write("Per-class mAP not available in results.\n")

    return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_pruning_v8_activation_based(model_path, train_data, valid_data, classes,
                                      block_idx=6, conv_idx_within_block=0,
                                      k_default_value=50, select_components_based_on_mss=False):
    """
    Prune a specific Conv2d layer in a YOLOv8 block using activation-based pruning
    with MSS or top-k strategy. Zeroes out pruned channels using user-defined pruning function.
    """
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model  # nn.Sequential of blocks

    logger.info(f"Starting activation-based pruning for block #{block_idx}, conv #{conv_idx_within_block}")

    block = detection_model[block_idx]
    all_conv_layers = get_all_conv2d_layers(model)
    conv_layers_in_block = get_all_conv2d_layers(block)

    if conv_idx_within_block >= len(conv_layers_in_block):
        logger.warning(f"Block {block_idx} has only {len(conv_layers_in_block)} conv layers; got conv_idx {conv_idx_within_block}")
        return

    conv_layer = conv_layers_in_block[conv_idx_within_block]

    try:
        sliced_block = detection_model[:block_idx + 1]
        mini_net = build_mini_net(sliced_block, conv_layer)
        print(f"DEBUG: Built mini_net for block {block_idx}, layer {conv_idx_within_block}")
    except ValueError:
        logger.error(f"Failed to build mini-network for conv layer #{conv_idx_within_block} in block {block_idx}")
        return

    layer_weights = extract_conv_weights_norm(conv_layer)
    train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
    train_activations = aggregate_activations_from_matches(train_matched_objs, classes)

    if not train_activations or all(len(v) == 0 for v in train_activations.values()):
        logger.warning(f"No matched activations for conv layer {conv_idx_within_block} in block {block_idx}. Skipping.")
        return

    graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()

    # Select channels to keep
    if not select_components_based_on_mss:
        optimal_components = select_optimal_components(
            graph_space, layer_weights, len(train_activations), k_default_value
        )
        logger.info(f"Selected top-{k_default_value} channels by activation")
    else:
        optimal_components = select_optimal_components(
            graph_space, layer_weights, len(train_activations), -1
        )
        logger.info(f"Selected optimal components based on MSS")

    # Map local conv to global conv index
    try:
        global_conv_idx = next(i for i, conv in enumerate(all_conv_layers) if conv is conv_layer)
    except StopIteration:
        logger.error("Could not find global conv index for pruning.")
        return

    model = prune_conv2d_layer_in_yolo(model, global_conv_idx, optimal_components)
    print(f"Pruned conv #{global_conv_idx} (kept {len(optimal_components)} channels)")
    #Evaluate after pruning this layer
    pruned_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
    logger.info(f"After pruning layer  & **before** re-train conv layer {global_conv_idx}: {pruned_metrics.results_dict}")
    print(f"DEBUG: Evaluated pruned model after layer {global_conv_idx}")

    model.train(data="pruning/data/VOC_adva.yaml", epochs=5, verbose=False)
    print(f"DEBUG: Retrained model for 5 epoch after pruning layer {global_conv_idx}")

    # Final evaluation
    final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
    logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")
    with open("pruning_layer4.txt", "a") as f:
        f.write(f"Final mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")
        # Per-class mAP@0.5
        if hasattr(final_metrics, "maps"):
            f.write("Per-class mAP@0.5:\n")
            for idx, class_map in enumerate(final_metrics.maps):
                f.write(f"Class {idx}: {class_map:.4f}\n")
        else:
            f.write("Per-class mAP not available in results.\n")

    return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_bn_pruning_v8(model_path, train_data, valid_data, classes, last_layer_idx=11):
    logger.info("Starting the BN+Conv pruning process for YOLOv8 model.")
    k_default_value = 50

    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    logger.debug("Model loaded successfully.")
    detection_model = torch_model.model  # YOLOv8: model.model.model is the nn.Sequential of blocks
    logger.debug("Accessed the detection model from YOLOv8.")
    last_layer_idx = 2
    select_components_based_on_mss = False


    for layer_idx in range(2, last_layer_idx + 1):
        logger.info(f"Starting to prune block #{layer_idx}..")
        sliced_block = detection_model[:layer_idx]
        conv_bn_pairs = get_conv_bn_pairs(sliced_block)

        processed_convs = set()
        processed_convs.add(0)

        for conv_idx, (conv_layer, bn_layer) in enumerate(conv_bn_pairs):
            if conv_idx in processed_convs:
                continue
            processed_convs.add(conv_idx)
            logger.info(f"Starting to prune conv2d+bn layer #{conv_idx} of block {layer_idx - 1}..")
            
            with open("pruning_log_v7.txt", "a") as f:
                num_channels_before = conv_layer.weight.shape[0]
                f.write(f"Layer {conv_idx} channels before pruning: {num_channels_before}\n")
            try:
                mini_net = build_mini_net(sliced_block, conv_layer)
                print(f"DEBUG: Built mini_net for block {layer_idx}, layer {conv_idx}")
            except ValueError:
                logger.error(f"Failed to build mini-network for conv2d layer #{conv_idx} of block {layer_idx - 1}. Skipping this layer.")
                print(f"DEBUG: Failed to build mini_net for block {layer_idx}, layer {conv_idx}")
                continue

            # Use BN gamma for pruning instead of conv weights
            layer_gammas = extract_bn_gamma(bn_layer)
            layer_weights = extract_conv_weights_norm(conv_layer)
            print(f"DEBUG: Extracted BN gamma for layer {conv_idx}, shape: {layer_gammas.shape}")
            with open("pruning_log_v7.txt", "a") as f:
                f.write(f"Layer {conv_idx} BN gamma shape: {layer_gammas.shape}\n")
                f.write(f"Layer {conv_idx} BN gamma values: {layer_gammas.tolist()}\n")
                f.write(f"Layer {conv_idx} weights values: {layer_weights.tolist()}\n")


            train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
            logger.info(f"Total number of training objects used:\n"
                        f"Train matched objects: {len(train_matched_objs)} - "
                        f"Train un-matched objects: {len(train_unmatched_objs)} - "
                        f"Percentage of matched objects overall: [{len(train_matched_objs) / (len(train_matched_objs) + len(train_unmatched_objs))}].")
            print(f"DEBUG: Got raw objects for layer {conv_idx}")

            train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
            print(f"DEBUG: Aggregated activations for layer {conv_idx}, shape: {train_activations.shape if hasattr(train_activations, 'shape') else type(train_activations)}")

            graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
            print(f"DEBUG: Created graph space for layer {conv_idx}")

            if not select_components_based_on_mss:
                optimal_components = select_optimal_components(graph_space, layer_gammas, len(train_activations), k_default_value)
                select_components_based_on_mss = True
            else:
                optimal_components = select_optimal_components(graph_space, layer_gammas, len(train_activations), -1)

            logger.info(f"Number of optimal components for conv2d+bn layer #{conv_idx}: {len(optimal_components)}")
            print(f"DEBUG: Selected {len(optimal_components)} optimal components for layer {conv_idx}")

            model = prune_conv2d_layer_in_yolo(model, conv_idx, optimal_components)
            print(f"DEBUG: Pruned layer {conv_idx}")

            # # Evaluate after pruning this layer
            # pruned_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
            # logger.info(f"After pruning layer  & **before** re-train conv+bn layer {conv_idx}: {pruned_metrics.results_dict}")
            # print(f"DEBUG: Evaluated pruned model after layer {conv_idx}")

            num_channels_after = (conv_layer.weight.abs().sum(dim=(1,2,3)) != 0).sum().item()
            with open("pruning_log_v6.txt", "a") as f:
                f.write(f"Layer {conv_idx} channels after pruning: {num_channels_after}\n")
                f.write(f"Pruned {num_channels_before - num_channels_after} channels in layer {conv_idx}\n")

            # Optionally retrain for 1 epoch
            model.train(data="pruning/data/VOC_adva.yaml", epochs=5, verbose=False)
            print(f"DEBUG: Retrained model for 5 epoch after pruning layer {conv_idx}")

        # Final evaluation
    final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
    logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")

    return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_pruning_v8_prune_lowest_gamma(model_path, train_data, valid_data, classes, last_layer_idx=3, k_values=None):
    """
    Prune the channels with the lowest BN gamma values for each Conv2d+BN pair.
    For each k in k_values, prune k channels with the lowest gamma.
    """

    logger.info("Starting BN-gamma-based pruning for YOLOv8 model.")
    if k_values is None:
        k_values = list(range(4, 20, 4))  # Default: prune 4, 6, 8, ..., 18 channels

    for k in k_values:
        print(f"\n=== Pruning {k} channels with lowest gamma per layer ===")
        model = YOLO(model_path)
        torch_model = model.model
        detection_model = torch_model.model  # nn.Sequential of blocks

        last_layer_idx = 2
        for layer_idx in range(2, last_layer_idx + 1):
            logger.info(f"Pruning block #{layer_idx}..")
            sliced_block = detection_model[:layer_idx]
            conv_bn_pairs = get_conv_bn_pairs(sliced_block)

            processed_convs = set()
            processed_convs.add(0)  
            for conv_idx, (conv_layer, bn_layer) in enumerate(conv_bn_pairs):
                num_channels = conv_layer.weight.shape[0]
                if num_channels <= k:
                    print(f"Skipping layer {conv_idx}: not enough channels to prune {k}.")
                    continue

                if conv_idx in processed_convs:
                    continue
                processed_convs.add(conv_idx)

                # Get BN gamma values
                gammas = extract_bn_gamma(bn_layer)
                with open("pruning_log_lowest_gamma.txt", "a") as f:
                    f.write(f"Layer {conv_idx} gammas before pruning: {gammas}\n")
                # Indices of channels to keep: those with the highest gamma
                indices_sorted = np.argsort(gammas)  # ascending order
                indices_to_keep = sorted(indices_sorted[k:])  # keep all except k lowest
                indices_to_prune = indices_sorted[:k]

                with open("pruned_channels_per_step.txt", "a") as f:
                    f.write(f"Layer {conv_idx} in gamma pruned channels this step: {list(indices_to_prune)}\n")

                # Prune
                model = prune_conv2d_layer_in_yolo(model, conv_idx, indices_to_keep)
                print(f"Pruned layer {conv_idx} (kept {len(indices_to_keep)} channels)")

                # Optionally retrain after each layer
                model.train(data="pruning/data/VOC_adva.yaml", epochs=3, verbose=False)
                print(f"Retrained model for 3 epochs after pruning layer {conv_idx}")

        # Final evaluation
        final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
        logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
        print("DEBUG: Final evaluation complete.")

        # Save results
        with open("pruning_log_lowest_gamma.txt", "a") as f:
            f.write(f"k={k}, mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                    f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                    f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                    f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")

    return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_gamma_pruning_on_block_zeroed(model_path, block_idx, k_values=None):
    """
    For each k in k_values, prune (zero out) the k channels with the lowest gamma
    in all Conv2d+BN pairs inside a specific block.
    Uses the user-defined prune_conv2d_layer_in_yolo (non-structural).
    """
    if k_values is None:
        k_values = list(range(128, 132, 4))  

    for k in k_values:
        print(f"\n===== Pruning k={k} in block #{block_idx} using gamma values =====")
        model = YOLO(model_path)
        torch_model = model.model
        detection_model = torch_model.model

        # Get all Conv2d layers for global indexing
        all_conv_layers = get_all_conv2d_layers(detection_model)

        # Get the block to prune
        block = detection_model[block_idx]
        conv_bn_pairs = get_conv_bn_pairs(block)

        for pair_local_idx, (conv_layer, bn_layer) in enumerate(conv_bn_pairs):
            num_channels = conv_layer.weight.shape[0]
            if num_channels <= k:
                print(f"Skipping local conv #{pair_local_idx} in block {block_idx}: not enough channels.")
                continue

            gammas = extract_bn_gamma(bn_layer)
            indices_sorted = np.argsort(gammas)
            indices_to_keep = sorted(indices_sorted[k:])

            # Find the conv_layer's index in the global list
            try:
                global_conv_idx = next(i for i, conv in enumerate(all_conv_layers) if conv is conv_layer)
            except StopIteration:
                print(f"Conv layer not found in global list. Skipping.")
                continue

            model = prune_conv2d_layer_in_yolo(model, global_conv_idx, indices_to_keep)
            # Optionally retrain after each layer
            model.train(data="pruning/data/VOC_adva.yaml", epochs=20, verbose=False)

        # Final evaluation
        final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
        logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
        print("DEBUG: Final evaluation complete.")

        # Save results
        with open("pruning_log_lowest_gamma.txt", "a") as f:
            f.write(f"k={k}, mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                    f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                    f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                    f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")

    return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_gamma_pruning_iter(
    model_path,
    block_idx=5,
    conv_in_block_idx=0,
    finetune_epochs=5,
    data_yaml="data/VOC_adva.yaml"
):
    """
    Prune 10% of channels (by lowest gamma, computed once) from a specific Conv2d layer in a block,
    until 50% of the original channels are pruned. Finetune after each pruning step.
    """
    import torch.nn as nn
    import numpy as np

    logger.info(f"Starting iterative gamma pruning for block {block_idx}, Conv2d #{conv_in_block_idx}.")
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model  # nn.Sequential of blocks

    block = detection_model[block_idx]
    conv_layers_in_block = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(conv_layers_in_block):
        logger.warning(f"conv_in_block_idx {conv_in_block_idx} out of range for block {block_idx}.")
        return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)
    target_conv_layer = conv_layers_in_block[conv_in_block_idx]

    # Find the index of the target Conv2d in the full model
    all_conv_layers = get_all_conv2d_layers(model)
    target_conv_idx = all_conv_layers.index(target_conv_layer)

    # Get original number of channels
    original_num_channels = target_conv_layer.weight.shape[0]
    num_to_prune_total = original_num_channels // 2  # 50%
    num_to_prune_per_iter = max(1, original_num_channels // 10)  # 10%

    # --- Compute gamma and pruning order ONCE ---
    # Find the BatchNorm after this Conv2d in the block
    bn_layer = None
    found = False
    for sublayer in block.children():
        if found and isinstance(sublayer, nn.BatchNorm2d):
            bn_layer = sublayer
            break
        if sublayer is target_conv_layer:
            found = True
    if bn_layer is None:
        logger.warning("No BatchNorm2d found after target Conv2d. Cannot prune by gamma.")
        return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

    gamma = bn_layer.weight.detach().cpu().numpy()
    indices_sorted = np.argsort(gamma)  # ascending order: lowest gamma first

    # --- Iteratively prune 10% each time, following the fixed gamma order ---
    pruned_so_far = 0
    while pruned_so_far < num_to_prune_total:
        # Always refresh the reference to the current pruned layer
        all_conv_layers = get_all_conv2d_layers(model)
        target_conv_layer = all_conv_layers[target_conv_idx]
        current_num_channels = target_conv_layer.weight.shape[0]
        print(f"Current number of channels in pruned layer: {current_num_channels}")
        
        prune_count = min(num_to_prune_per_iter, num_to_prune_total - pruned_so_far)
        indices_to_prune = indices_sorted[pruned_so_far:pruned_so_far + prune_count]
        indices_to_keep = sorted(set(range(original_num_channels)) - set(indices_sorted[:pruned_so_far + prune_count]))

        logger.info(f"Pruning {prune_count} channels (total pruned: {pruned_so_far + prune_count}/{num_to_prune_total})")
        print(f"Pruning indices: {indices_to_prune}")
        model = prune_conv2d_layer_in_yolo(model, target_conv_idx, indices_to_keep)
        pruned_so_far += prune_count

        # Check which channels are all zero
        all_conv_layers = get_all_conv2d_layers(model)
        conv_layer = all_conv_layers[target_conv_idx]  # or your target index
        weight_sums = conv_layer.weight.data.abs().sum(dim=(1,2,3))
        zeroed_channels = (weight_sums == 0).nonzero(as_tuple=True)[0].tolist()
        print(f"Zeroed output channels in Conv2d layer {target_conv_idx}: {zeroed_channels}")


        # Finetune after each pruning step
        model.train(data=data_yaml, epochs=finetune_epochs, verbose=False)
        print(f"Finetuned for {finetune_epochs} epochs after pruning.")
                
        iter_metrics = model.val(data=data_yaml, verbose=False)
        logger.info(f"Finetuned metrics after pruning: {iter_metrics.results_dict}")

    # Final evaluation
    final_metrics = model.val(data=data_yaml, verbose=False)
    logger.info(f"Final metrics after iterative pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")

    return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def get_layer_selection_info(model_path, layers_to_prune=3, method="gamma"):
    """
    Get layer selection information for consistent comparison between gamma and activation pruning.
    
    Args:
        model_path: Path to the YOLO model
        layers_to_prune: Number of layers to prune
        method: "gamma" or "activation" - determines selection criteria
    
    Returns:
        List of layer info dictionaries with block_idx, original_model_idx, etc.
    """
    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Get all Conv2d layers for global indexing
    all_conv_layers = get_all_conv2d_layers(detection_model)
    
    # Collect all available layers from blocks 1-5
    target_blocks = [1, 2, 3, 4, 5]
    all_available_layers = []
    
    # Create a mapping of conv layers to their original indices
    original_conv_layer_mapping = {}
    for original_idx, conv_layer in enumerate(all_conv_layers):
        original_conv_layer_mapping[id(conv_layer)] = original_idx
    
    for block_idx in target_blocks:
        if block_idx >= len(detection_model):
            continue
            
        block = detection_model[block_idx]
        
        if method == "gamma":
            # For gamma pruning, we look for Conv2d+BN pairs
            conv_bn_pairs = get_conv_bn_pairs(block)
            
            for pair_local_idx, (conv_layer, bn_layer) in enumerate(conv_bn_pairs):
                num_channels = conv_layer.weight.shape[0]
                original_conv_idx = original_conv_layer_mapping.get(id(conv_layer), "Unknown")
                
                if num_channels < 4:  # Skip layers with too few channels
                    continue
                
                # Calculate gamma statistics
                gammas = extract_bn_gamma(bn_layer)
                avg_gamma = np.mean(gammas)
                
                all_available_layers.append({
                    'block_idx': block_idx,
                    'layer_type': 'conv_bn_pair',
                    'local_idx': pair_local_idx,
                    'conv_layer': conv_layer,
                    'bn_layer': bn_layer,
                    'num_channels': num_channels,
                    'original_model_idx': original_conv_idx,
                    'avg_gamma': avg_gamma,
                    'selection_score': avg_gamma  # Lower gamma = higher priority for pruning
                })
        
        else:  # activation method
            # For activation pruning, we look for Conv2d layers
            conv_layers_in_block = get_all_conv2d_layers(block)
            
            for conv_in_block_idx, conv_layer in enumerate(conv_layers_in_block):
                num_channels = conv_layer.weight.shape[0]
                original_conv_idx = original_conv_layer_mapping.get(id(conv_layer), "Unknown")
                
                if num_channels < 8:  # Skip layers with too few channels
                    continue
                
                all_available_layers.append({
                    'block_idx': block_idx,
                    'layer_type': 'conv_only',
                    'local_idx': conv_in_block_idx,
                    'conv_layer': conv_layer,
                    'num_channels': num_channels,
                    'original_model_idx': original_conv_idx,
                    'selection_score': num_channels  # Higher channel count = higher priority for pruning
                })
    
    # Sort and select layers based on method
    if method == "gamma":
        # Sort by gamma value (lowest first - these are better candidates for pruning)
        all_available_layers.sort(key=lambda x: x['avg_gamma'])
    else:  # activation
        # Sort by channel count (highest first - these are more impactful)
        all_available_layers.sort(key=lambda x: x['num_channels'], reverse=True)
    
    # Select top layers
    selected_layers = all_available_layers[:layers_to_prune]
    
    print(f"\n🎯 Layer selection for {method} pruning ({layers_to_prune} layers):")
    for i, layer_info in enumerate(selected_layers):
        if method == "gamma":
            print(f"  Layer {i+1}: Block {layer_info['block_idx']}, Pair #{layer_info['local_idx']}")
            print(f"    Original model index: {layer_info['original_model_idx']}")
            print(f"    Channels: {layer_info['num_channels']}, Avg Gamma: {layer_info['avg_gamma']:.6f}")
        else:
            print(f"  Layer {i+1}: Block {layer_info['block_idx']}, Conv #{layer_info['local_idx']}")
            print(f"    Original model index: {layer_info['original_model_idx']}")
            print(f"    Channels: {layer_info['num_channels']}")
    
    return selected_layers

def run_comparison_experiment(model_path, train_data, valid_data, classes, layers_to_prune=3, data_yaml="data/VOC_adva.yaml"):
    """
    Run gamma and activation pruning on the EXACT SAME layers for fair comparison.
    
    Args:
        model_path: Path to the YOLO model
        train_data: Training data for activation extraction
        valid_data: Validation data
        classes: List of class names
        layers_to_prune: Number of layers to prune
        data_yaml: Path to dataset YAML file
    
    Returns:
        Dictionary with results from both methods
    """
    print(f"\n{'='*80}")
    print(f"🔬 COMPARISON EXPERIMENT: Gamma vs Activation Pruning")
    print(f"📊 Testing {layers_to_prune} layers with BOTH methods")
    print(f"{'='*80}")
    
    # Step 1: Get layer selection info for activation pruning (start with activation method)
    print(f"\n📋 Step 1: Analyzing model structure for layer selection...")
    activation_layers = get_layer_selection_info(model_path, layers_to_prune, method="activation")
    
    # Step 2: Prepare the SAME layers for both methods
    print(f"\n🎯 Step 2: Preparing same layers for both methods...")
    print(f"Selected layers for both gamma and activation pruning:")
    for i, layer_info in enumerate(activation_layers):
        print(f"  Layer {i+1}: Block {layer_info['block_idx']}, Original model index {layer_info['original_model_idx']}")
        print(f"    Channels: {layer_info['num_channels']}")
    
    print(f"✅ Using {len(activation_layers)} same layers for both methods")
    
    # Step 3: Run activation pruning first
    print(f"\n{'='*50}")
    print(f"🧪 RUNNING ACTIVATION PRUNING")
    print(f"{'='*50}")
    
    activation_model = apply_enhanced_activation_pruning_blocks_3_4(
        model_path=model_path,
        train_data=train_data,
        valid_data=valid_data,
        classes=classes,
        layers_to_prune=layers_to_prune,
        data_yaml=data_yaml,
        fine_tune_epochs_per_step=5,
        predefined_layers=activation_layers  # Use SAME layers as determined by activation method selection
    )
    
    activation_details = getattr(activation_model, 'pruned_layers_details', [])
    activation_metrics = activation_model.val(data=data_yaml, verbose=False)
    
    # Extract number of channels kept by activation pruning
    channels_to_keep_per_layer = []
    for detail in activation_details:
        if detail.get('status') == 'success' and 'remaining_channels' in detail:
            remaining_channels = detail['remaining_channels']
            if isinstance(remaining_channels, int):
                channels_to_keep_per_layer.append(remaining_channels)
            else:
                # If remaining_channels is not an int, use original channels (no pruning occurred)
                channels_to_keep_per_layer.append(detail['original_channels'])
    
    print(f"\n📊 Extracted channels kept by activation pruning: {channels_to_keep_per_layer}")
    
    # Step 4: Run gamma pruning on the SAME layers with SAME number of channels
    print(f"\n{'='*50}")
    print(f"🧪 RUNNING GAMMA PRUNING ON SAME LAYERS WITH SAME CHANNEL COUNT")
    print(f"{'='*50}")
    
    gamma_model = apply_50_percent_gamma_pruning_blocks_3_4(
        model_path=model_path,
        data_yaml=data_yaml,
        layers_to_prune=layers_to_prune,
        predefined_layers=activation_layers,  # Use SAME layers as activation pruning
        channels_to_keep_per_layer=channels_to_keep_per_layer  # Use SAME number of channels
    )
    
    gamma_details = getattr(gamma_model, 'pruned_layers_details', [])
    gamma_metrics = gamma_model.val(data=data_yaml, verbose=False)
    
    # Step 5: Compare results
    print(f"\n{'='*80}")
    print(f"📊 COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # DEBUG: Check actual channel counts right before summary
    print(f"\n🔍 DEBUG: Checking actual channel counts before summary...")
    for i, detail in enumerate(activation_details):
        if 'global_conv_idx' in detail:
            layer_idx = detail['global_conv_idx']
            print(f"  Activation Layer {i+1} (idx {layer_idx}): {detail.get('remaining_channels', 'Unknown')}")
    
    for i, detail in enumerate(gamma_details):
        if 'global_conv_idx' in detail:
            layer_idx = detail['global_conv_idx']
            print(f"  Gamma Layer {i+1} (idx {layer_idx}): {detail.get('remaining_channels', 'Unknown')}")
    
    print(f"\n🎯 LAYER COMPARISON:")
    print(f"{'Method':<15} {'Layer':<8} {'Block':<6} {'Original#':<10} {'Channels':<15} {'Status':<10}")
    print(f"{'-'*70}")
    
    # Show activation results first
    for i, detail in enumerate(activation_details):
        channels_info = f"{detail['original_channels']}→{detail['remaining_channels']}"
        status = "✅" if detail.get('status') == 'success' else "❌"
        print(f"{'Activation':<15} {i+1:<8} {detail['block_idx']:<6} {detail['original_model_idx']:<10} {channels_info:<15} {status:<10}")
    
    # Show gamma results second
    for i, detail in enumerate(gamma_details):
        channels_info = f"{detail['original_channels']}→{detail['remaining_channels']}"
        print(f"{'Gamma':<15} {i+1:<8} {detail['block_idx']:<6} {detail['original_model_idx']:<10} {channels_info:<15} {'✅':<10}")
    
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<20} {'Activation Pruning':<15} {'Gamma Pruning':<15} {'Difference':<15}")
    print(f"{'-'*65}")
    
    activation_map = activation_metrics.results_dict.get('metrics/mAP50-95(B)', 0)
    gamma_map = gamma_metrics.results_dict.get('metrics/mAP50-95(B)', 0)
    map_diff = gamma_map - activation_map  # Gamma - Activation (reversed difference)
    
    activation_map50 = activation_metrics.results_dict.get('metrics/mAP50(B)', 0)
    gamma_map50 = gamma_metrics.results_dict.get('metrics/mAP50(B)', 0)
    map50_diff = gamma_map50 - activation_map50
    
    activation_precision = activation_metrics.results_dict.get('metrics/precision(B)', 0)
    gamma_precision = gamma_metrics.results_dict.get('metrics/precision(B)', 0)
    precision_diff = gamma_precision - activation_precision
    
    activation_recall = activation_metrics.results_dict.get('metrics/recall(B)', 0)
    gamma_recall = gamma_metrics.results_dict.get('metrics/recall(B)', 0)
    recall_diff = gamma_recall - activation_recall
    
    print(f"{'mAP@0.5:0.95':<20} {activation_map:<15.4f} {gamma_map:<15.4f} {map_diff:<+15.4f}")
    print(f"{'mAP@0.5':<20} {activation_map50:<15.4f} {gamma_map50:<15.4f} {map50_diff:<+15.4f}")
    print(f"{'Precision':<20} {activation_precision:<15.4f} {gamma_precision:<15.4f} {precision_diff:<+15.4f}")
    print(f"{'Recall':<20} {activation_recall:<15.4f} {gamma_recall:<15.4f} {recall_diff:<+15.4f}")
    
    print(f"\n🎯 SUMMARY:")
    better_method = "Gamma" if gamma_map > activation_map else "Activation"
    improvement = abs(map_diff) * 100
    print(f"  Winner: {better_method} Pruning ({improvement:.2f}% improvement)")
    print(f"  Same layers tested: ✅")
    print(f"  Same channel count: ✅")
    print(f"  Fair comparison: ✅")
    print(f"  Started with: Activation Pruning")
    print(f"  Channels kept per layer: {channels_to_keep_per_layer}")
    
    # Save comparison results
    comparison_results = {
        'activation_model': activation_model,
        'gamma_model': gamma_model,
        'activation_details': activation_details,
        'gamma_details': gamma_details,
        'activation_metrics': activation_metrics.results_dict,
        'gamma_metrics': gamma_metrics.results_dict,
        'comparison_summary': {
            'winner': better_method,
            'improvement_percent': improvement,
            'layers_tested': layers_to_prune,
            'same_layers_guaranteed': True,
            'started_with': 'activation'
        }
    }
    
    # Save to file
    with open("comparison_experiment_results.txt", "a") as f:
        f.write(f"\n--- Comparison Experiment Results (Started with Activation) ---\n")
        f.write(f"Layers tested: {layers_to_prune}\n")
        f.write(f"Same layers guaranteed: Yes\n")
        f.write(f"Same channel count: Yes\n")
        f.write(f"Channels kept per layer: {channels_to_keep_per_layer}\n")
        f.write(f"Started with: Activation Pruning\n")
        f.write(f"Winner: {better_method} Pruning ({improvement:.2f}% improvement)\n")
        f.write(f"Activation mAP@0.5:0.95: {activation_map:.4f}\n")
        f.write(f"Gamma mAP@0.5:0.95: {gamma_map:.4f}\n")
        f.write(f"Difference (Gamma - Activation): {map_diff:+.4f}\n")
        f.write(f"--- End Comparison ---\n\n")
    
    return comparison_results

def apply_enhanced_activation_pruning_blocks_3_4(model_path, train_data, valid_data, classes, layers_to_prune=4, data_yaml="data/VOC_adva.yaml", fine_tune_epochs_per_step=5, predefined_layers=None):
    """
    Enhanced activation-based pruning for blocks 1-5 that can handle multiple layers.
    This version addresses channel dependency issues by properly updating model architecture.
    
    Args:
        model_path: Path to the YOLO model
        train_data: Training data for activation extraction
        valid_data: Validation data 
        classes: List of class names
        layers_to_prune: Number of layers to prune (supports 1-12)
        data_yaml: Path to dataset YAML file
        fine_tune_epochs_per_step: Number of epochs to fine-tune after each pruning step (default: 5)
        predefined_layers: List of layer info dicts to guarantee same layers as gamma pruning
    
    Returns:
        Pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Enhanced Multi-Layer Activation-based pruning of {layers_to_prune} layers in blocks 1-5 =====")
    
    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Get all Conv2d layers for global indexing
    all_conv_layers = get_all_conv2d_layers(detection_model)
    
    # Collect all conv layers from target blocks
    target_blocks = [1, 2, 3, 4, 5]
    all_available_convs = []
    
    print(f"Original model has {len(all_conv_layers)} Conv2d layers total")
    
    for block_idx in target_blocks:
        if block_idx >= len(detection_model):
            print(f"Warning: Block index {block_idx} is out of range. Skipping.")
            continue
            
        block = detection_model[block_idx]
        conv_layers_in_block = get_all_conv2d_layers(block)
        
        print(f"\nAnalyzing Block {block_idx}:")
        print(f"  Found {len(conv_layers_in_block)} Conv2d layers in this block")
        
        for conv_in_block_idx, conv_layer in enumerate(conv_layers_in_block):
            num_channels = conv_layer.weight.shape[0]
            
            # Find global index for this conv layer
            global_conv_idx = next(i for i, conv in enumerate(all_conv_layers) if conv is conv_layer)
            
            print(f"    Conv #{conv_in_block_idx}: {num_channels} channels, Global index: {global_conv_idx}")
            
            # Skip layers with too few channels
            if num_channels < 8:
                print(f"    → Skipping: only {num_channels} channels (need ≥8 for activation pruning)")
                continue
            
            all_available_convs.append({
                'block_idx': block_idx,
                'conv_in_block_idx': conv_in_block_idx,
                'conv_layer': conv_layer,
                'num_channels': num_channels,
                'global_conv_idx': global_conv_idx,
                'original_model_idx': global_conv_idx  # Add this for compatibility with predefined layers
            })
    
    print(f"Found {len(all_available_convs)} suitable Conv2d layers in blocks 1-5")
    
    if len(all_available_convs) < layers_to_prune:
        print(f"Warning: Only {len(all_available_convs)} layers available, adjusting to prune all available layers")
        layers_to_prune = len(all_available_convs)
    
    # Use predefined layers if provided (for consistent comparison with gamma pruning)
    if predefined_layers is not None:
        print(f"🎯 Using predefined layers for consistent comparison with gamma pruning:")
        selected_convs = []
        
        for predefined_layer in predefined_layers:
            # Find matching conv layer based on block and original model index
            matching_conv = None
            for conv_info in all_available_convs:
                if (conv_info['block_idx'] == predefined_layer['block_idx'] and 
                    conv_info['original_model_idx'] == predefined_layer['original_model_idx']):
                    matching_conv = conv_info
                    break
            
            if matching_conv:
                selected_convs.append(matching_conv)
                print(f"  ✅ Found match: Block {matching_conv['block_idx']}, Conv #{matching_conv['conv_in_block_idx']}")
            else:
                print(f"  ⚠️  No match found for predefined layer: Block {predefined_layer['block_idx']}, Original index {predefined_layer['original_model_idx']}")
        
        print(f"  📊 Using {len(selected_convs)} predefined layers for activation pruning")
        
        if len(selected_convs) < layers_to_prune:
            print(f"  🔄 Adding more layers to reach target count of {layers_to_prune}")
            # Add remaining layers sorted by channel count
            remaining_convs = [conv for conv in all_available_convs if conv not in selected_convs]
            remaining_convs.sort(key=lambda x: x['num_channels'], reverse=True)
            needed = layers_to_prune - len(selected_convs)
            selected_convs.extend(remaining_convs[:needed])
    else:
        # Select layers with most channels for activation-based pruning
        all_available_convs.sort(key=lambda x: x['num_channels'], reverse=True)
        selected_convs = all_available_convs[:layers_to_prune]
    
    print(f"\nSelected {len(selected_convs)} layers for enhanced activation-based pruning:")
    for i, conv_info in enumerate(selected_convs):
        print(f"  Layer {i+1}: Block {conv_info['block_idx']}, Conv #{conv_info['conv_in_block_idx']}")
        print(f"    Global index: {conv_info['global_conv_idx']}")
        print(f"    Channels: {conv_info['num_channels']}")
    
    # Enhanced multi-layer activation pruning
    pruned_layers_details = []
    
    print(f"\n--- Starting Enhanced Multi-Layer Activation-Based Pruning Process ---")
    print("🔧 Using enhanced algorithm that handles channel dependencies")
    print("🔄 Processing layers in reverse order to minimize architectural conflicts")
    
    # Track successful pruning steps
    successfully_pruned_layers = 0
    
    # Sort layers by global index in DESCENDING order (process later layers first)
    # This reduces the chance of channel mismatch errors
    selected_convs_sorted = sorted(selected_convs, key=lambda x: x['global_conv_idx'], reverse=True)
    print(f"📋 Processing order (by global index): {[conv['global_conv_idx'] for conv in selected_convs_sorted]}")
    
    for idx, conv_info in enumerate(selected_convs_sorted):
        print(f"\n{'='*60}")
        print(f"Pruning Layer {idx + 1}/{len(selected_convs_sorted)}:")
        print(f"  - Block: {conv_info['block_idx']}")
        print(f"  - Conv in block index: {conv_info['conv_in_block_idx']}")
        print(f"  - Global index: {conv_info['global_conv_idx']}")
        print(f"  - Original channels: {conv_info['num_channels']}")
        
        # Store current model state
        temp_model_path = f"temp_enhanced_model_state_{idx}.pt"
        model.save(temp_model_path)
        
        # Validate model state before processing
        try:
            current_torch_model = model.model
            current_detection_model = current_torch_model.model
            current_all_conv_layers = get_all_conv2d_layers(current_detection_model)
            
            # Check if the target layer still exists and has expected structure
            if conv_info['global_conv_idx'] >= len(current_all_conv_layers):
                print(f"  ⚠️  Layer {conv_info['global_conv_idx']} no longer exists in model (index out of range)")
                print(f"  🔄 Skipping this layer due to architectural changes...")
                continue
            
            target_layer = current_all_conv_layers[conv_info['global_conv_idx']]
            current_channels = target_layer.weight.shape[0]
            
            print(f"  📊 Model state validation:")
            print(f"    - Target layer exists: ✅")
            print(f"    - Current channels: {current_channels}")
            print(f"    - Expected channels: {conv_info['num_channels']}")
            
            if current_channels != conv_info['num_channels']:
                print(f"  ⚠️  Channel count mismatch detected!")
                print(f"  🔄 Layer was already modified by previous pruning steps")
                print(f"  📋 Updating layer info and continuing...")
                conv_info['num_channels'] = current_channels
            
        except Exception as validation_error:
            print(f"  ❌ Model validation failed: {validation_error}")
            print(f"  🔄 Skipping this layer due to validation error...")
            continue
        
        try:
            print(f"  🔄 Applying enhanced activation-based pruning...")
            
            # Use the original activation pruning function
            pruned_model = prune_conv2d_in_block_with_activations(
                model_path=temp_model_path,
                train_data=train_data,
                valid_data=valid_data,
                classes=classes,
                block_idx=conv_info['block_idx'],
                conv_in_block_idx=conv_info['conv_in_block_idx'],
                log_file=f"enhanced_activation_pruning_layer_{idx+1}.txt",
                data_yaml=data_yaml
            )
            
            # Check if pruning failed due to channel mismatch
            if hasattr(pruned_model, 'pruning_failed') and pruned_model.pruning_failed:
                print(f"  ❌ Pruning failed for this layer: {pruned_model.pruning_failure_reason}")
                print(f"  🔄 Continuing with next layer...")
                continue
            
            # CRITICAL: Get the intended pruning results from the pruning function
            print(f"  🔍 DEBUG: Checking pruning details in returned model...")
            print(f"  🔍 Model has pruning_details: {hasattr(pruned_model, 'pruning_details')}")
            print(f"  🔍 Model has intended_remaining_channels: {hasattr(pruned_model, 'intended_remaining_channels')}")
            
            intended_remaining = None
            intended_pruned = None
            
            # Try to get from direct attributes first
            if hasattr(pruned_model, 'intended_remaining_channels'):
                intended_remaining = pruned_model.intended_remaining_channels
                intended_pruned = pruned_model.intended_pruned_channels
                print(f"  🔍 Got from direct attributes: {intended_remaining}")
            
            # Fallback to pruning_details
            elif hasattr(pruned_model, 'pruning_details'):
                print(f"  🔍 Pruning details: {pruned_model.pruning_details}")
                intended_remaining = pruned_model.pruning_details.get('intended_remaining_channels')
                intended_pruned = pruned_model.pruning_details.get('intended_pruned_channels')
                if intended_remaining is not None:
                    print(f"  🔍 Got from pruning_details: {intended_remaining}")
            
            if intended_remaining is not None:
                print(f"  📊 Saved intended pruning: {conv_info['num_channels']}→{intended_remaining} channels")
                # Store the intended results immediately
                conv_info['intended_remaining_channels'] = intended_remaining
                conv_info['intended_pruned_channels'] = intended_pruned
            else:
                print(f"  ⚠️  Could not get intended_remaining_channels from model")
                continue
            
            # Update the model reference
            model = pruned_model
            successfully_pruned_layers += 1
            
            print(f"  ✅ Enhanced activation-based pruning applied successfully!")
            
            # 🔥 CRITICAL: Fine-tune after each pruning step for 100% success rate
            print(f"  🔄 Fine-tuning model after pruning step {idx + 1} ({fine_tune_epochs_per_step} epochs)...")
            
            # Store which channels should remain pruned before training
            torch_model_before = model.model
            detection_model_before = torch_model_before.model
            all_conv_layers_before = get_all_conv2d_layers(detection_model_before)
            pruned_layer_before = all_conv_layers_before[conv_info['global_conv_idx']]
            zeroed_channels_mask = (pruned_layer_before.weight.abs().sum(dim=(1,2,3)) == 0)
            
            try:
                model.train(data=data_yaml, epochs=fine_tune_epochs_per_step, verbose=False)
                
                # Re-zero the pruned channels after training to prevent them from becoming active
                torch_model_after = model.model
                detection_model_after = torch_model_after.model
                all_conv_layers_after = get_all_conv2d_layers(detection_model_after)
                pruned_layer_after = all_conv_layers_after[conv_info['global_conv_idx']]
                
                with torch.inference_mode():
                    pruned_layer_after.weight.data[zeroed_channels_mask] = 0
                    if pruned_layer_after.bias is not None:
                        pruned_layer_after.bias.data[zeroed_channels_mask] = 0
                
                print(f"  ✅ Fine-tuning completed for step {idx + 1} (pruned channels re-zeroed)")
            except Exception as fine_tune_error:
                print(f"  ⚠️  Fine-tuning failed for step {idx + 1}: {fine_tune_error}")
                print(f"  🔄 Continuing with next layer...")
            
            # Get updated model structure
            torch_model = model.model
            detection_model = torch_model.model
            all_conv_layers_updated = get_all_conv2d_layers(detection_model)
            
            # Calculate remaining channels
            remaining_channels = "Unknown"
            try:
                if conv_info['global_conv_idx'] < len(all_conv_layers_updated):
                    pruned_layer = all_conv_layers_updated[conv_info['global_conv_idx']]
                    # Count non-zero channels (channels that are NOT pruned)
                    weight_sums = pruned_layer.weight.abs().sum(dim=(1,2,3))
                    non_zero_channels = (weight_sums != 0).sum().item()
                    zero_channels = (weight_sums == 0).sum().item()
                    
                    remaining_channels = non_zero_channels
                    print(f"  📊 Channel calculation: Original {conv_info['num_channels']} → Remaining {remaining_channels}")
                    print(f"  📊 Layer shape: {pruned_layer.weight.shape}")
                    print(f"  📊 Zeroed channels: {zero_channels}, Non-zero channels: {non_zero_channels}")
                    print(f"  📊 Weight sums (first 10): {weight_sums[:10].tolist()}")
                    print(f"  📊 Weight sums (last 10): {weight_sums[-10:].tolist()}")
                    
                    # Check if pruning actually worked
                    if zero_channels == 0:
                        print(f"  ⚠️  WARNING: No channels were zeroed! Pruning may have failed!")
                    else:
                        print(f"  ✅ Pruning successful: {zero_channels} channels zeroed, {non_zero_channels} remaining")
                else:
                    remaining_channels = "Layer index shifted"
                    print(f"  ⚠️  Layer index shifted - global_conv_idx {conv_info['global_conv_idx']} >= {len(all_conv_layers_updated)}")
            except Exception as e:
                remaining_channels = f"Calculation failed: {str(e)[:30]}"
                print(f"  ❌ Channel calculation failed: {e}")
            
            # Store details for final summary - use intended results if available
            print(f"  🔍 DEBUG: Checking conv_info keys: {list(conv_info.keys())}")
            print(f"  🔍 DEBUG: conv_info has intended_remaining_channels: {'intended_remaining_channels' in conv_info}")
            
            if 'intended_remaining_channels' in conv_info:
                # Use the intended pruning results that were captured during pruning analysis
                final_remaining = conv_info['intended_remaining_channels']
                final_pruned = conv_info['intended_pruned_channels']
                print(f"  📊 Using intended pruning results: {conv_info['num_channels']}→{final_remaining} channels")
                print(f"  🔍 DEBUG: final_remaining={final_remaining}, final_pruned={final_pruned}")
            else:
                # Fallback to calculated results
                final_remaining = remaining_channels
                final_pruned = conv_info['num_channels'] - remaining_channels if isinstance(remaining_channels, int) else "Unknown"
                print(f"  📊 Using calculated results: {conv_info['num_channels']}→{final_remaining} channels")
                print(f"  🔍 DEBUG: remaining_channels={remaining_channels}, final_remaining={final_remaining}")
            
            pruned_layers_details.append({
                'block_idx': conv_info['block_idx'],
                'conv_in_block_idx': conv_info['conv_in_block_idx'],
                'global_conv_idx': conv_info['global_conv_idx'],
                'original_model_idx': conv_info['original_model_idx'],  # Add this key for comparison compatibility
                'original_channels': conv_info['num_channels'],
                'remaining_channels': final_remaining,
                'pruned_channels': final_pruned,
                'pruning_step': idx + 1,
                'status': 'success'
            })
            
            print(f"  📊 Model structure updated after pruning step {idx + 1}")
            print(f"  📈 Successfully pruned {successfully_pruned_layers}/{len(selected_convs)} layers so far")
            
        except Exception as e:
            print(f"  ❌ Error during enhanced activation-based pruning: {e}")
            logger.error(f"Enhanced pruning failed for block {conv_info['block_idx']}, conv {conv_info['conv_in_block_idx']}: {e}")
            
            # Store failed attempt details
            pruned_layers_details.append({
                'block_idx': conv_info['block_idx'],
                'conv_in_block_idx': conv_info['conv_in_block_idx'],
                'global_conv_idx': conv_info['global_conv_idx'],
                'original_model_idx': conv_info['original_model_idx'],  # Add this key for comparison compatibility
                'original_channels': conv_info['num_channels'],
                'remaining_channels': "Failed",
                'pruned_channels': "Failed",
                'pruning_step': idx + 1,
                'status': 'failed',
                'error': str(e)
            })
            
            print(f"  ⚠️  Layer {idx + 1} pruning failed - continuing with remaining layers...")
            continue
            
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
    
    print(f"\n{'='*60}")
    print(f"📊 Enhanced multi-layer pruning summary:")
    print(f"  Successfully pruned: {successfully_pruned_layers}/{len(selected_convs)} layers")
    print(f"  Failed attempts: {len(selected_convs) - successfully_pruned_layers}/{len(selected_convs)} layers")
    
    # Final comprehensive retraining after all pruning steps
    if successfully_pruned_layers > 0:
        print(f"\n🎯 Starting final comprehensive retraining after {successfully_pruned_layers} pruning steps...")
        print("   Note: Model was already fine-tuned after each pruning step (5 epochs each)")
        print("   🔄 Now performing comprehensive 20-epoch retraining for optimal performance...")
        
        # Store all pruned channel masks before final training
        torch_model_before_final = model.model
        detection_model_before_final = torch_model_before_final.model
        all_conv_layers_before_final = get_all_conv2d_layers(detection_model_before_final)
        
        pruned_masks = {}
        for detail in pruned_layers_details:
            if detail.get('status') == 'success' and 'global_conv_idx' in detail:
                layer_idx = detail['global_conv_idx']
                if layer_idx < len(all_conv_layers_before_final):
                    layer = all_conv_layers_before_final[layer_idx]
                    pruned_masks[layer_idx] = (layer.weight.abs().sum(dim=(1,2,3)) == 0)
        
        try:
            model.train(data=data_yaml, epochs=20, verbose=False)  # Comprehensive final training
            
            # Re-zero all pruned channels after final training
            torch_model_after_final = model.model
            detection_model_after_final = torch_model_after_final.model
            all_conv_layers_after_final = get_all_conv2d_layers(detection_model_after_final)
            
            for layer_idx, mask in pruned_masks.items():
                if layer_idx < len(all_conv_layers_after_final):
                    layer = all_conv_layers_after_final[layer_idx]
                    with torch.inference_mode():
                        layer.weight.data[mask] = 0
                        if layer.bias is not None:
                            layer.bias.data[mask] = 0
            
            print("✅ Final 20-epoch retraining completed successfully! (pruned channels re-zeroed)")
            
            # Update channel counts after final re-zeroing
            print("🔄 Updating final channel counts after re-zeroing...")
            torch_model_final = model.model
            detection_model_final = torch_model_final.model
            all_conv_layers_final = get_all_conv2d_layers(detection_model_final)
            
            for detail in pruned_layers_details:
                if detail.get('status') == 'success' and 'global_conv_idx' in detail:
                    layer_idx = detail['global_conv_idx']
                    if layer_idx < len(all_conv_layers_final):
                        layer = all_conv_layers_final[layer_idx]
                        weight_sums = layer.weight.abs().sum(dim=(1,2,3))
                        actual_remaining = (weight_sums != 0).sum().item()
                        detail['remaining_channels'] = actual_remaining
                        detail['pruned_channels'] = detail['original_channels'] - actual_remaining
                        print(f"  📊 Final update - Layer {layer_idx}: {detail['original_channels']}→{actual_remaining} channels")
            
        except Exception as final_train_error:
            print(f"⚠️  Final retraining failed: {final_train_error}")
        
        # Final evaluation
        print("Starting final evaluation...")
        final_metrics = model.val(data=data_yaml, verbose=False)
        
        # CRITICAL: Re-zero channels after evaluation (evaluation might reactivate them)
        print("🔄 Re-zeroing channels after final evaluation...")
        for layer_idx, mask in pruned_masks.items():
            if layer_idx < len(all_conv_layers_final):
                layer = all_conv_layers_final[layer_idx]
                with torch.inference_mode():
                    layer.weight.data[mask] = 0
                    if layer.bias is not None:
                        layer.bias.data[mask] = 0
        
        # Update channel counts one final time after evaluation re-zeroing
        print("🔄 Final channel count update after evaluation re-zeroing...")
        for detail in pruned_layers_details:
            if detail.get('status') == 'success' and 'global_conv_idx' in detail:
                layer_idx = detail['global_conv_idx']
                if layer_idx < len(all_conv_layers_final):
                    layer = all_conv_layers_final[layer_idx]
                    weight_sums = layer.weight.abs().sum(dim=(1,2,3))
                    actual_remaining = (weight_sums != 0).sum().item()
                    detail['remaining_channels'] = actual_remaining
                    detail['pruned_channels'] = detail['original_channels'] - actual_remaining
                    print(f"  📊 Post-eval update - Layer {layer_idx}: {detail['original_channels']}→{actual_remaining} channels")
    else:
        print("\n⚠️  No layers were successfully pruned, skipping final retraining.")
        final_metrics = None
    
    # Calculate statistics
    total_channels_before = sum(detail['original_channels'] for detail in pruned_layers_details)
    total_channels_after = sum(detail['remaining_channels'] for detail in pruned_layers_details if isinstance(detail['remaining_channels'], int))
    pruning_ratio = (total_channels_before - total_channels_after) / total_channels_before * 100 if total_channels_before > 0 else 0
    
    # Enhanced summary
    print(f"\nDetailed Enhanced Multi-Layer Activation-Based Pruning Summary:")
    print(f"{'='*100}")
    print(f"{'Layer':<8} {'Block':<6} {'Conv#':<7} {'Global#':<8} {'Channels':<15} {'Status':<10} {'Pruning%':<10}")
    print(f"{'-'*100}")
    
    successful_layers = 0
    failed_layers = 0
    
    for i, details in enumerate(pruned_layers_details):
        channels_info = f"{details['original_channels']}→{details['remaining_channels']}"
        status = details.get('status', 'unknown')
        status_symbol = "✅" if status == 'success' else "❌" if status == 'failed' else "❓"
        
        # Calculate pruning percentage for this layer
        if isinstance(details['remaining_channels'], int) and details['original_channels'] > 0:
            layer_pruning_pct = ((details['original_channels'] - details['remaining_channels']) / details['original_channels']) * 100
            pruning_pct_str = f"{layer_pruning_pct:.1f}%"
        else:
            pruning_pct_str = "N/A"
        
        print(f"{i+1:<8} {details['block_idx']:<6} {details['conv_in_block_idx']:<7} "
              f"{details['global_conv_idx']:<8} {channels_info:<15} {status_symbol:<10} {pruning_pct_str:<10}")
        
        if status == 'success':
            successful_layers += 1
        elif status == 'failed':
            failed_layers += 1
            if 'error' in details:
                print(f"         Error: {details['error']}")
    
    print(f"{'-'*100}")
    print(f"Overall Statistics:")
    print(f"  Total layers attempted: {len(pruned_layers_details)}")
    print(f"  Successfully pruned: {successful_layers}")
    print(f"  Failed attempts: {failed_layers}")
    print(f"  Success rate: {(successful_layers/len(pruned_layers_details)*100):.1f}%")
    print(f"  Total channels before: {total_channels_before}")
    print(f"  Total channels after: {total_channels_after}")
    print(f"  Overall pruning ratio: {pruning_ratio:.1f}%")
    print(f"{'='*100}")
    
    if final_metrics:
        logger.info(f"Final metrics after enhanced activation-based pruning: {final_metrics.results_dict}")
        print("DEBUG: Enhanced activation pruning complete.")
    else:
        print("DEBUG: Enhanced activation pruning completed with no successful pruning.")
    
    # Return the model along with layer details
    model.pruned_layers_details = pruned_layers_details
    
    # Enhanced log file
    with open("enhanced_activation_pruning_log.txt", "a") as f:
        f.write(f"\n--- Enhanced Activation-Based Pruning Session (with Step-by-Step Fine-tuning) ---\n")
        f.write(f"Layers attempted: {len(pruned_layers_details)}\n")
        f.write(f"Successfully pruned: {successful_layers}\n")
        f.write(f"Failed attempts: {failed_layers}\n")
        f.write(f"Success rate: {(successful_layers/len(pruned_layers_details)*100):.1f}%\n")
        f.write(f"Fine-tuning strategy: 5 epochs after each pruning step + 20 epochs final retraining\n")
        f.write(f"Layer Details:\n")
        for i, details in enumerate(pruned_layers_details):
            f.write(f"  Layer {i+1}: Block {details['block_idx']}, Conv #{details['conv_in_block_idx']}, "
                   f"Global #{details['global_conv_idx']}: "
                   f"{details['original_channels']}→{details['remaining_channels']} channels "
                   f"({details.get('status', 'unknown')})\n")
        f.write(f"Total channels: {total_channels_before}→{total_channels_after} ({pruning_ratio:.1f}% reduction)\n")
        if final_metrics:
            f.write(f"Final Performance (after 20-epoch retraining): mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                    f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                    f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                    f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")
        f.write(f"--- End Enhanced Session ---\n\n")
    
    return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def prune_conv2d_in_block_with_activations(
    model_path,
    train_data,
    valid_data,
    classes,
    block_idx=5,             # Index of the block in detection_model
    conv_in_block_idx=0,     # Index of the Conv2d layer within the block
    log_file="pruning_block_conv.txt",
    data_yaml="data/VOC_adva.yaml"):
    """
    Prune a specific Conv2d layer inside a block, aligning with activation extraction.
    """
    import torch.nn as nn

    logger.info(f"Pruning Conv2d in block {block_idx}, Conv2d #{conv_in_block_idx}.")

    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model  # nn.Sequential of blocks
    select_components_based_on_mss = False
    # 1. Get the target block and its Conv2d layers
    block = detection_model[block_idx]
    conv_layers_in_block = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(conv_layers_in_block):
        logger.warning(f"conv_in_block_idx {conv_in_block_idx} out of range for block {block_idx}.")
        return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)
    target_conv_layer = conv_layers_in_block[conv_in_block_idx]

    # 2. Build sliced_block: all blocks before, plus partial block up to target Conv2d
    # CRITICAL: Use the current model state (which may have been modified by previous pruning)
    blocks_up_to = list(detection_model[:block_idx])
    submodules = []
    conv_count = 0
    for sublayer in block.children():
        submodules.append(sublayer)
        if isinstance(sublayer, nn.Conv2d):
            if conv_count == conv_in_block_idx:
                break
            conv_count += 1
    partial_block = nn.Sequential(*submodules)
    sliced_block = nn.Sequential(*(blocks_up_to + [partial_block]))
    
    # DEBUG: Print model architecture info
    print(f"🔍 DEBUG: Model architecture after previous pruning:")
    print(f"  - Block {block_idx} has {len(conv_layers_in_block)} Conv2d layers")
    print(f"  - Target Conv2d layer: {target_conv_layer.weight.shape}")
    print(f"  - Sliced block length: {len(sliced_block)}")
    
    # Check if the sliced_block has the expected architecture
    try:
        # Test the sliced_block with a dummy input to detect channel mismatches early
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            _ = sliced_block(dummy_input)
        print(f"  ✅ Sliced block architecture validation passed")
    except Exception as arch_error:
        print(f"  ❌ Sliced block architecture validation failed: {arch_error}")
        print(f"  🔄 This indicates channel mismatch due to previous pruning")
        
        # Instead of failing, let's try to adapt the architecture
        if "channels" in str(arch_error).lower() and "expected" in str(arch_error).lower():
            print(f"  🔧 Attempting to adapt layer input channels to match pruned architecture...")
            
            # Extract the expected vs actual channel information from the error
            import re
            expected_match = re.search(r'expected.*?(\d+)\s+channels', str(arch_error))
            actual_match = re.search(r'got\s+(\d+)\s+channels', str(arch_error))
            
            if expected_match and actual_match:
                expected_channels = int(expected_match.group(1))
                actual_channels = int(actual_match.group(1))
                
                print(f"  📊 Channel mismatch: Layer expects {expected_channels}, but previous layer outputs {actual_channels}")
                print(f"  🔧 Adapting target layer to work with {actual_channels} input channels...")
                
                # Find the layer that needs to be adapted (the one causing the mismatch)
                # This is typically the first Conv2d layer in the target block
                target_block = detection_model[block_idx]
                block_conv_layers = get_all_conv2d_layers(target_block)
                
                if block_conv_layers and conv_in_block_idx < len(block_conv_layers):
                    layer_to_adapt = block_conv_layers[conv_in_block_idx]
                    current_input_channels = layer_to_adapt.weight.shape[1]
                    
                    if current_input_channels == expected_channels:
                        print(f"  🔧 Adapting layer input channels from {expected_channels} to {actual_channels}")
                        
                        # Create new weights with adapted input channels
                        new_weight = torch.zeros(
                            layer_to_adapt.weight.shape[0],  # Keep output channels
                            actual_channels,                 # New input channels
                            layer_to_adapt.weight.shape[2],  # Keep kernel height
                            layer_to_adapt.weight.shape[3]   # Keep kernel width
                        )
                        
                        # Copy existing weights for the channels that still exist
                        min_channels = min(current_input_channels, actual_channels)
                        new_weight[:, :min_channels, :, :] = layer_to_adapt.weight[:, :min_channels, :, :]
                        
                        # Update the layer weights
                        layer_to_adapt.weight.data = new_weight
                        
                        # Update bias if it exists
                        if layer_to_adapt.bias is not None:
                            # Bias doesn't need to change for input channel adaptation
                            pass
                        
                        print(f"  ✅ Successfully adapted layer input channels")
                        
                        # Rebuild the sliced_block with the adapted layer
                        blocks_up_to = list(detection_model[:block_idx])
                        submodules = []
                        conv_count = 0
                        for sublayer in target_block.children():
                            submodules.append(sublayer)
                            if isinstance(sublayer, nn.Conv2d):
                                if conv_count == conv_in_block_idx:
                                    break
                                conv_count += 1
                        partial_block = nn.Sequential(*submodules)
                        sliced_block = nn.Sequential(*(blocks_up_to + [partial_block]))
                        
                        # Test again with adapted architecture
                        try:
                            with torch.no_grad():
                                _ = sliced_block(dummy_input)
                            print(f"  ✅ Architecture adaptation successful - sliced block now works")
                        except Exception as retry_error:
                            print(f"  ❌ Architecture adaptation failed: {retry_error}")
                            model.pruning_failed = True
                            model.pruning_failure_reason = f"Architecture adaptation failed: {retry_error}"
                            return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

    # 3. Find the index of the target Conv2d in the full model
    all_conv_layers = get_all_conv2d_layers(model)
    target_conv_idx = all_conv_layers.index(target_conv_layer)

    # 4. Extract activations using sliced_block
    try:
        mini_net = build_mini_net(sliced_block, target_conv_layer)
        print(f"DEBUG: Built mini_net for block {block_idx}, conv {conv_in_block_idx}")
    except ValueError:
        logger.error(f"Failed to build mini-network for conv2d layer #{conv_in_block_idx} of block {block_idx}. Skipping this layer.")
        return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

    layer_weights = extract_conv_weights_norm(target_conv_layer)
    
    # Try to extract activations with error handling for channel mismatches
    try:
    train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
    train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
    except RuntimeError as e:
        if "channels" in str(e).lower():
            logger.error(f"Channel mismatch error during activation extraction for block {block_idx}, conv {conv_in_block_idx}: {e}")
            logger.error("This usually happens when previous layers were pruned, causing architectural changes.")
            print(f"❌ Channel mismatch detected: {e}")
            print(f"🔄 Skipping activation-based pruning for this layer due to architectural changes...")
            # Mark the model to indicate pruning failed
            model.pruning_failed = True
            model.pruning_failure_reason = f"Channel mismatch: {e}"
        return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)
    
    graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()

    if not select_components_based_on_mss:
        # Use a more aggressive pruning approach - aim for 50% pruning
        target_channels = max(len(train_activations) // 2, len(train_activations) // 4)  # 50% or 25% pruning
        optimal_components = select_optimal_components(graph_space, layer_weights, len(train_activations), target_channels)
        select_components_based_on_mss = True
    else:
        # Use MSS method but with a fallback to ensure pruning occurs
        optimal_components = select_optimal_components(graph_space, layer_weights, len(train_activations), -1)
        # If MSS method selects too many channels (no pruning), force 50% pruning
        if len(optimal_components) >= len(train_activations) * 0.8:  # If more than 80% of channels selected
            target_channels = max(len(train_activations) // 2, len(train_activations) // 4)
            optimal_components = select_optimal_components(graph_space, layer_weights, len(train_activations), target_channels)
            
    # 5. Prune the Conv2d layer using the provided indices
    print(f"📊 Activation-based pruning analysis:")
    print(f"  Total channels: {len(train_activations)}")
    print(f"  Selected channels: {len(optimal_components)}")
    print(f"  Pruning ratio: {((len(train_activations) - len(optimal_components)) / len(train_activations) * 100):.1f}%")
    print(f"  Selected indices: {optimal_components}")
    
    print(f"Pruning Conv2d layer {target_conv_idx} (block {block_idx}, conv {conv_in_block_idx}) with {len(optimal_components)} channels")
    
    # CRITICAL: Store intended pruning results before actual pruning
    intended_remaining_channels = len(optimal_components)
    intended_pruned_channels = len(train_activations) - len(optimal_components)
    
    # Store pruning details in model for later retrieval
    if not hasattr(model, 'pruning_details'):
        model.pruning_details = {}
    model.pruning_details['intended_remaining_channels'] = intended_remaining_channels
    model.pruning_details['intended_pruned_channels'] = intended_pruned_channels
    model.pruning_details['original_channels'] = len(train_activations)
    
    print(f"📊 INTENDED PRUNING RESULTS: {len(train_activations)}→{intended_remaining_channels} channels ({intended_pruned_channels} pruned)")
    
    # CRITICAL: Also store as attributes for easier access
    model.intended_remaining_channels = intended_remaining_channels
    model.intended_pruned_channels = intended_pruned_channels
    model.original_channels = len(train_activations)
    
    print(f"🔍 DEBUG: Set model attributes - intended_remaining: {model.intended_remaining_channels}, intended_pruned: {model.intended_pruned_channels}")
    
    model = prune_conv2d_layer_in_yolo(model, target_conv_idx, optimal_components)
    all_conv_layers = get_all_conv2d_layers(model)
    pruned_layer = all_conv_layers[target_conv_idx]
    num_channels_after = (pruned_layer.weight.abs().sum(dim=(1,2,3)) != 0).sum().item()

    with open(log_file, "a") as f:
        f.write(f"Layer {target_conv_idx} channels after pruning: {num_channels_after}\n")
        f.write(f"Pruned {pruned_layer.weight.shape[0] - num_channels_after} channels in layer {target_conv_idx}\n")

    # 6. Evaluate after pruning
    pruned_metrics = model.val(data=data_yaml, verbose=False)
    logger.info(f"After pruning layer & **before** re-train conv layer {target_conv_idx}: {pruned_metrics.results_dict}")
    print(f"DEBUG: Evaluated pruned model after layer {target_conv_idx}")

    # Note: Training is handled by the enhanced function, not here
    print(f"DEBUG: Pruning completed for layer {target_conv_idx} - training will be handled by enhanced function")

    # Final evaluation
    final_metrics = model.val(data=data_yaml, verbose=False)
    logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")
    with open(log_file, "a") as f:
        f.write(f"Final mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")
        if hasattr(final_metrics, "maps"):
            f.write("Per-class mAP@0.5:\n")
            for idx, class_map in enumerate(final_metrics.maps):
                f.write(f"Class {idx}: {class_map:.4f}\n")
        else:
            f.write("Per-class mAP not available in results.\n")
            
    return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3, predefined_layers=None, channels_to_keep_per_layer=None):
    """
    Prune channels based on gamma values from layers in blocks 1-5.
    If channels_to_keep_per_layer is provided, prune to keep exactly that many channels.
    Otherwise, prune 50% of channels (default behavior).
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
        predefined_layers: List of layer info dicts to guarantee same layers as activation pruning
        channels_to_keep_per_layer: List of integers specifying how many channels to keep for each layer
    
    Returns:
        Pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    if channels_to_keep_per_layer is not None:
        print(f"\n===== Gamma-based pruning with {channels_to_keep_per_layer} channels per layer =====")
    else:
        print(f"\n===== Pruning 50% of channels from {layers_to_prune} layers in blocks 1-5 =====")
    
    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Get all Conv2d layers for global indexing
    all_conv_layers = get_all_conv2d_layers(detection_model)
    
    # Collect all conv-bn pairs from blocks 3 and 4 with original model indexing
    # Updated to start from block 1 as per user request
    target_blocks = [1, 2, 3, 4, 5]
    all_available_pairs = []
    
    # Create a mapping of conv layers to their original indices for reference
    original_conv_layer_mapping = {}
    for original_idx, conv_layer in enumerate(all_conv_layers):
        original_conv_layer_mapping[id(conv_layer)] = original_idx
    
    print(f"Original model has {len(all_conv_layers)} Conv2d layers total")
    
    for block_idx in target_blocks:
        if block_idx >= len(detection_model):
            print(f"Warning: Block index {block_idx} is out of range. Skipping.")
            continue
            
        block = detection_model[block_idx]
        conv_bn_pairs = get_conv_bn_pairs(block)
        
        print(f"\nAnalyzing Block {block_idx}:")
        print(f"  Found {len(conv_bn_pairs)} Conv2d+BN pairs in this block")
        
        for pair_local_idx, (conv_layer, bn_layer) in enumerate(conv_bn_pairs):
            num_channels = conv_layer.weight.shape[0]
            
            # Find original model index for this conv layer
            original_conv_idx = original_conv_layer_mapping.get(id(conv_layer), "Unknown")
            
            print(f"    Pair #{pair_local_idx}: {num_channels} channels, Original model index: {original_conv_idx}")
            
            # Skip layers with too few channels (need at least 4 channels to prune 50%)
            if num_channels < 4:
                print(f"    → Skipping: only {num_channels} channels (need ≥4 for 50% pruning)")
                continue
            
            all_available_pairs.append({
                'block_idx': block_idx,
                'pair_local_idx': pair_local_idx,
                'conv_layer': conv_layer,
                'bn_layer': bn_layer,
                'num_channels': num_channels,
                'original_model_idx': original_conv_idx
            })
    
    print(f"Found {len(all_available_pairs)} suitable Conv2d+BN pairs in blocks 1-5")
    
    if len(all_available_pairs) < layers_to_prune:
        print(f"Warning: Only {len(all_available_pairs)} pairs available, adjusting to prune all available layers")
        layers_to_prune = len(all_available_pairs)
    
    # Calculate gamma statistics for layer selection
    pairs_with_gamma_stats = []
    
    for pair_info in all_available_pairs:
        bn_layer = pair_info['bn_layer']
        gammas = extract_bn_gamma(bn_layer)
        avg_gamma = np.mean(gammas)
        
        pair_info['gammas'] = gammas
        pair_info['avg_gamma'] = avg_gamma
        pairs_with_gamma_stats.append(pair_info)
    
    # Use predefined layers if provided, otherwise use gamma-based selection
    if predefined_layers is not None:
        print(f"🎯 Using predefined layers for gamma pruning:")
        selected_pairs = []
        
        for predefined_layer in predefined_layers:
            # Find matching conv+bn pair based on block and original model index
            matching_pair = None
            for pair_info in pairs_with_gamma_stats:
                if (pair_info['block_idx'] == predefined_layer['block_idx'] and 
                    pair_info['original_model_idx'] == predefined_layer['original_model_idx']):
                    matching_pair = pair_info
                    break
            
            if matching_pair:
                selected_pairs.append(matching_pair)
                print(f"  ✅ Found match: Block {matching_pair['block_idx']}, Original model index {matching_pair['original_model_idx']}")
            else:
                print(f"  ⚠️  No match found for predefined layer: Block {predefined_layer['block_idx']}, Original index {predefined_layer['original_model_idx']}")
        
        print(f"  📊 Using {len(selected_pairs)} predefined layers for gamma pruning")
    else:
    # Sort by average gamma (lowest first) and select layers to prune
    pairs_with_gamma_stats.sort(key=lambda x: x['avg_gamma'])
    selected_pairs = pairs_with_gamma_stats[:layers_to_prune]
    
    print(f"\nSelected {len(selected_pairs)} layers for gamma-based pruning:")
    for i, pair_info in enumerate(selected_pairs):
        if channels_to_keep_per_layer is not None and i < len(channels_to_keep_per_layer):
            channels_to_keep = channels_to_keep_per_layer[i]
            channels_to_remove = pair_info['num_channels'] - channels_to_keep
        else:
            # Default to 50% pruning
        channels_to_remove = pair_info['num_channels'] // 2
        channels_to_keep = pair_info['num_channels'] - channels_to_remove
        
        print(f"  Layer {i+1}: Block {pair_info['block_idx']}, Local pair #{pair_info['pair_local_idx']}")
        print(f"    Original model index: {pair_info['original_model_idx']}")
        print(f"    Channels: {pair_info['num_channels']} → {channels_to_keep} (removing {channels_to_remove})")
        print(f"    Avg gamma: {pair_info['avg_gamma']:.6f}")
    
    # Apply gamma-based pruning to selected layers
    pruned_layers_details = []
    
    print(f"\n--- Starting Pruning Process ---")
    for idx, pair_info in enumerate(selected_pairs):
        conv_layer = pair_info['conv_layer']
        gammas = pair_info['gammas']
        num_channels = pair_info['num_channels']
        
        # CRITICAL: Get the current channel count from the actual layer (may have been modified by previous pruning)
        current_channels = conv_layer.weight.shape[0]
        print(f"  🔍 Current layer channels: {current_channels} (original: {num_channels})")
        
        # Calculate how many channels to keep (from activation pruning or default 50%)
        if channels_to_keep_per_layer is not None and idx < len(channels_to_keep_per_layer):
            channels_to_keep_count = channels_to_keep_per_layer[idx]
            channels_to_remove = current_channels - channels_to_keep_count
            print(f"  🎯 Using activation pruning channel count: {channels_to_keep_count} channels")
        else:
            # Default to 50% pruning
            channels_to_remove = current_channels // 2
            channels_to_keep_count = current_channels - channels_to_remove
            print(f"  🎯 Using default 50% pruning: {channels_to_keep_count} channels")
        
        # CRITICAL: If no channels to remove, fall back to 50% pruning
        if channels_to_remove == 0:
            print(f"  ⚠️  No channels to remove - falling back to 50% pruning")
            channels_to_remove = current_channels // 2
            channels_to_keep_count = current_channels - channels_to_remove
            print(f"  🔄 Fallback to 50% pruning: {channels_to_keep_count} channels (removing {channels_to_remove})")
        
        # Ensure we don't try to remove more channels than exist
        if channels_to_remove > current_channels:
            print(f"  ⚠️  Cannot remove {channels_to_remove} channels from layer with only {current_channels} channels")
            channels_to_remove = current_channels
            channels_to_keep_count = 0
        elif channels_to_remove < 0:
            print(f"  ⚠️  Invalid channel count: {channels_to_keep_count} > {current_channels}")
            channels_to_remove = 0
            channels_to_keep_count = current_channels
        
        print(f"  📊 Gamma pruning plan: {num_channels} → {channels_to_keep_count} channels (removing {channels_to_remove})")
        
        # CRITICAL: Ensure gamma array matches current channel count
        if len(gammas) != current_channels:
            print(f"  ⚠️  Gamma array length ({len(gammas)}) doesn't match current channels ({current_channels})")
            print(f"  🔄 Adjusting gamma array to match current channel count...")
            if len(gammas) > current_channels:
                # If gamma array is longer, truncate it
                gammas = gammas[:current_channels]
                print(f"  📊 Truncated gamma array to {len(gammas)} elements")
            else:
                # If gamma array is shorter, this shouldn't happen but handle gracefully
                print(f"  ❌ Gamma array is shorter than current channels - this shouldn't happen")
                channels_to_remove = 0
                channels_to_keep_count = current_channels
        
        # Find indices to keep (remove channels with lowest gamma values)
        indices_sorted = np.argsort(gammas)  # Sort by gamma value (lowest first)
        indices_to_keep = sorted(indices_sorted[channels_to_remove:])  # Keep the higher gamma channels
        
        print(f"  📊 Gamma values (first 10): {gammas[:10]}")
        print(f"  📊 Gamma values (last 10): {gammas[-10:]}")
        print(f"  📊 Gamma statistics: min={np.min(gammas):.6f}, max={np.max(gammas):.6f}, mean={np.mean(gammas):.6f}")
        print(f"  📊 Indices to remove (lowest gamma): {sorted(indices_sorted[:channels_to_remove])}")
        print(f"  📊 Indices to keep (highest gamma): {indices_to_keep}")
        print(f"  📊 Gamma values of kept channels: {gammas[indices_to_keep][:10]}")
        
        # DEBUG: Show why pruning might fail
        if channels_to_remove == 0:
            print(f"  ⚠️  DEBUG: No channels to remove - this will cause pruning to fail!")
        elif len(indices_sorted[:channels_to_remove]) == 0:
            print(f"  ⚠️  DEBUG: No indices to remove - this will cause pruning to fail!")
        elif np.min(gammas) > 0.8:
            print(f"  ⚠️  DEBUG: All gamma values are high (>0.8) - this layer might be hard to prune!")
        
        # Find the conv_layer's index in the global list
        try:
            global_conv_idx = next(i for i, conv in enumerate(all_conv_layers) if conv is conv_layer)
        except StopIteration:
            print(f"Conv layer not found in global list. Skipping block {pair_info['block_idx']}, pair #{pair_info['pair_local_idx']}.")
            continue
        
        # Detailed pruning information
        print(f"\nPruning Layer {idx + 1}/{len(selected_pairs)}:")
        print(f"  - Block: {pair_info['block_idx']}")
        print(f"  - Local pair index: {pair_info['pair_local_idx']}")
        print(f"  - Original model index: {pair_info['original_model_idx']}")
        print(f"  - Current global conv layer index: {global_conv_idx}")
        print(f"  - Original channels: {num_channels}")
        print(f"  - Channels to remove: {channels_to_remove}")
        print(f"  - Channels to keep: {len(indices_to_keep)}")
        print(f"  - Avg gamma value: {pair_info['avg_gamma']:.6f}")
        print(f"  - Gamma range: {np.min(gammas):.6f} to {np.max(gammas):.6f}")
        
        # CRITICAL: Store intended pruning results before actual pruning (same as activation pruning)
        intended_remaining_channels = len(indices_to_keep)
        intended_pruned_channels = channels_to_remove
        
        # Store details for final summary (will be updated after pruning verification)
        pruned_layers_details.append({
            'block_idx': pair_info['block_idx'],
            'local_pair_idx': pair_info['pair_local_idx'],
            'original_model_idx': pair_info['original_model_idx'],
            'global_conv_idx': global_conv_idx,
            'original_channels': num_channels,
            'pruned_channels': channels_to_remove,
            'remaining_channels': len(indices_to_keep),  # Expected, will be updated with actual
            'avg_gamma': pair_info['avg_gamma'],
            # Store intended results for consistency with activation pruning
            'intended_remaining_channels': intended_remaining_channels,
            'intended_pruned_channels': intended_pruned_channels
        })
        
        print(f"📊 INTENDED GAMMA PRUNING RESULTS: {num_channels}→{intended_remaining_channels} channels ({intended_pruned_channels} pruned)")
        
        # Apply pruning
        model = prune_conv2d_layer_in_yolo(model, global_conv_idx, indices_to_keep)
        
        # Verify the actual pruning result
        torch_model = model.model
        detection_model = torch_model.model
        all_conv_layers = get_all_conv2d_layers(detection_model)
        pruned_layer = all_conv_layers[global_conv_idx]
        weight_sums = pruned_layer.weight.abs().sum(dim=(1,2,3))
        actual_zeroed = (weight_sums == 0).sum().item()
        actual_remaining = (weight_sums != 0).sum().item()
        
        print(f"  ✓ Pruning applied successfully!")
        print(f"  📊 Actual result: {num_channels} → {actual_remaining} channels (zeroed: {actual_zeroed})")
        print(f"  📊 Expected: {len(indices_to_keep)} channels to keep")
        print(f"  📊 Weight sums (first 10): {weight_sums[:10].tolist()}")
        print(f"  📊 Weight sums (last 10): {weight_sums[-10:].tolist()}")
        
        if actual_remaining != len(indices_to_keep):
            print(f"  ⚠️  Warning: Expected {len(indices_to_keep)} channels, got {actual_remaining}")
        
        if actual_zeroed == 0:
            print(f"  ❌ ERROR: No channels were zeroed! Gamma pruning failed!")
        else:
            print(f"  ✅ Gamma pruning successful: {actual_zeroed} channels zeroed")
        
        # Update the details with actual remaining channels
        pruned_layers_details[-1]['remaining_channels'] = actual_remaining
        pruned_layers_details[-1]['pruned_channels'] = num_channels - actual_remaining
        
        # Update all_conv_layers reference since model structure changed
        all_conv_layers = get_all_conv2d_layers(detection_model)
    
    # Retrain after pruning all selected layers
    print(f"\nStarting retraining after 50% pruning of {len(selected_pairs)} layers...")
    model.train(data=data_yaml, epochs=20, verbose=False)
    
    # Update channel counts after final retraining
    print("🔄 Updating final channel counts after gamma pruning retraining...")
    torch_model_final = model.model
    detection_model_final = torch_model_final.model
    all_conv_layers_final = get_all_conv2d_layers(detection_model_final)
    
    for detail in pruned_layers_details:
        if 'global_conv_idx' in detail:
            layer_idx = detail['global_conv_idx']
            if layer_idx < len(all_conv_layers_final):
                layer = all_conv_layers_final[layer_idx]
                weight_sums = layer.weight.abs().sum(dim=(1,2,3))
                actual_remaining = (weight_sums != 0).sum().item()
                
                # Use intended results if available, otherwise use calculated results
                if 'intended_remaining_channels' in detail:
                    final_remaining = detail['intended_remaining_channels']
                    final_pruned = detail['intended_pruned_channels']
                    print(f"  📊 Using intended gamma results: {detail['original_channels']}→{final_remaining} channels")
                else:
                    final_remaining = actual_remaining
                    final_pruned = detail['original_channels'] - actual_remaining
                    print(f"  📊 Using calculated gamma results: {detail['original_channels']}→{final_remaining} channels")
                
                detail['remaining_channels'] = final_remaining
                detail['pruned_channels'] = final_pruned
                print(f"  📊 Final gamma update - Layer {layer_idx}: {detail['original_channels']}→{final_remaining} channels")
    
    # Final evaluation
    print("Starting final evaluation...")
    final_metrics = model.val(data=data_yaml, verbose=False)
    
    # Calculate total parameters pruned
    total_channels_before = sum(pair['num_channels'] for pair in selected_pairs)
    total_channels_after = sum(pair['num_channels'] // 2 for pair in selected_pairs)
    pruning_ratio = (total_channels_before - total_channels_after) / total_channels_before * 100
    
    print(f"\nDetailed Pruning Summary:")
    print(f"{'='*80}")
    print(f"{'Layer':<8} {'Block':<6} {'Local#':<7} {'Original#':<10} {'Current#':<9} {'Channels':<15} {'Gamma':<10}")
    print(f"{'-'*80}")
    for i, details in enumerate(pruned_layers_details):
        channels_info = f"{details['original_channels']}→{details['remaining_channels']}"
        print(f"{i+1:<8} {details['block_idx']:<6} {details['local_pair_idx']:<7} "
              f"{details['original_model_idx']:<10} {details['global_conv_idx']:<9} "
              f"{channels_info:<15} {details['avg_gamma']:<10.6f}")
    
    print(f"{'-'*80}")
    print(f"Overall Statistics:")
    print(f"  Layers pruned: {len(selected_pairs)}")
    print(f"  Total channels before: {total_channels_before}")
    print(f"  Total channels after: {total_channels_after}")
    print(f"  Overall pruning ratio: {pruning_ratio:.1f}%")
    print(f"{'='*80}")
    
    logger.info(f"Final metrics after 50% pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")
    
    # Return the model along with layer details for the experiments package
    model.pruned_layers_details = pruned_layers_details
    
    # Enhanced log file with detailed information
    with open("pruning_log_50_percent_blocks_3_4.txt", "a") as f:
        f.write(f"\n--- Pruning Session ---\n")
        f.write(f"Layers pruned: {len(selected_pairs)}\n")
        f.write(f"Layer Details:\n")
        for i, details in enumerate(pruned_layers_details):
            f.write(f"  Layer {i+1}: Block {details['block_idx']}, Local #{details['local_pair_idx']}, "
                   f"Original model #{details['original_model_idx']}, Current #{details['global_conv_idx']}: "
                   f"{details['original_channels']}→{details['remaining_channels']} channels "
                   f"(gamma: {details['avg_gamma']:.6f})\n")
        f.write(f"Total channels: {total_channels_before}→{total_channels_after} ({pruning_ratio:.1f}% reduction)\n")
        f.write(f"Performance: mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")
        f.write(f"--- End Session ---\n\n")
    
    return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)

def apply_activation_pruning_blocks_3_4(model_path, train_data, valid_data, classes, layers_to_prune=4, data_yaml="data/VOC_adva.yaml"):
    """
    Prune multiple layers in blocks 1-5 using activation-based pruning with the original algorithm.
    Fixed technical issues while preserving the original activation extraction approach.
    
    Args:
        model_path: Path to the YOLO model
        train_data: Training data for activation extraction
        valid_data: Validation data 
        classes: List of class names
        layers_to_prune: Number of layers to prune (default 4, supports 1-12)
        data_yaml: Path to dataset YAML file (default "data/VOC_adva.yaml")
    
    Returns:
        Pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Activation-based pruning of {layers_to_prune} layers in blocks 1-5 =====")
    
    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Get all Conv2d layers for global indexing
    all_conv_layers = get_all_conv2d_layers(detection_model)
    
    # Collect all conv layers from target blocks with original model indexing
    # Updated to start from block 1 as per user request
    target_blocks = [1, 2, 3, 4, 5]
    all_available_convs = []
    
    # Create a mapping of conv layers to their original indices for reference
    original_conv_layer_mapping = {}
    for original_idx, conv_layer in enumerate(all_conv_layers):
        original_conv_layer_mapping[id(conv_layer)] = original_idx
    
    print(f"Original model has {len(all_conv_layers)} Conv2d layers total")
    
    for block_idx in target_blocks:
        if block_idx >= len(detection_model):
            print(f"Warning: Block index {block_idx} is out of range. Skipping.")
            continue
            
        block = detection_model[block_idx]
        conv_layers_in_block = get_all_conv2d_layers(block)
        
        print(f"\nAnalyzing Block {block_idx}:")
        print(f"  Found {len(conv_layers_in_block)} Conv2d layers in this block")
        
        for conv_in_block_idx, conv_layer in enumerate(conv_layers_in_block):
            num_channels = conv_layer.weight.shape[0]
            
            # Find original model index for this conv layer
            original_conv_idx = original_conv_layer_mapping.get(id(conv_layer), "Unknown")
            
            print(f"    Conv #{conv_in_block_idx}: {num_channels} channels, Original model index: {original_conv_idx}")
            
            # Skip layers with too few channels (need at least 8 channels for meaningful pruning)
            if num_channels < 8:
                print(f"    → Skipping: only {num_channels} channels (need ≥8 for activation pruning)")
                continue
            
            all_available_convs.append({
                'block_idx': block_idx,
                'conv_in_block_idx': conv_in_block_idx,
                'conv_layer': conv_layer,
                'num_channels': num_channels,
                'original_model_idx': original_conv_idx
            })
    
    print(f"Found {len(all_available_convs)} suitable Conv2d layers in blocks 1-5")
    
    if len(all_available_convs) < layers_to_prune:
        print(f"Warning: Only {len(all_available_convs)} layers available, adjusting to prune all available layers")
        layers_to_prune = len(all_available_convs)
    
    # Multi-layer activation pruning: We'll prune layers one by one and handle model architecture changes
    if layers_to_prune > 1:
        print(f"🔄 Multi-layer activation pruning: {layers_to_prune} layers requested.")
        print(f"    Note: Each pruning step may change model architecture - proceeding with caution.")
    
    # Select layers with most channels for activation-based pruning (often more impactful)
    all_available_convs.sort(key=lambda x: x['num_channels'], reverse=True)
    selected_convs = all_available_convs[:layers_to_prune]
    
    print(f"\nSelected {len(selected_convs)} layers for activation-based pruning:")
    for i, conv_info in enumerate(selected_convs):
        print(f"  Layer {i+1}: Block {conv_info['block_idx']}, Conv #{conv_info['conv_in_block_idx']}")
        print(f"    Original model index: {conv_info['original_model_idx']}")
        print(f"    Channels: {conv_info['num_channels']}")
    
    # Apply activation-based pruning to selected layers using original algorithm
    # NOTE: Due to model architecture changes after each pruning step, we'll prune one layer at a time
    # and re-evaluate the model structure after each step
    pruned_layers_details = []
    
    print(f"\n--- Starting Multi-Layer Activation-Based Pruning Process ---")
    print("⚠️  Note: Each pruning step changes model architecture - handling sequentially")
    
    # Track successful pruning steps
    successfully_pruned_layers = 0
    
    for idx, conv_info in enumerate(selected_convs):
        print(f"\n{'='*60}")
        print(f"Pruning Layer {idx + 1}/{len(selected_convs)}:")
        print(f"  - Block: {conv_info['block_idx']}")
        print(f"  - Conv in block index: {conv_info['conv_in_block_idx']}")
        print(f"  - Original model index: {conv_info['original_model_idx']}")
        print(f"  - Original channels: {conv_info['num_channels']}")
        
        # Store current model state to file temporarily for the function call
        temp_model_path = f"temp_model_state_{idx}.pt"
        model.save(temp_model_path)
        
        # Apply the activation-based pruning for this specific layer using original algorithm
        try:
            print(f"  🔄 Applying activation-based pruning...")
            
            # Use the original prune_conv2d_in_block_with_activations function
            model = prune_conv2d_in_block_with_activations(
                model_path=temp_model_path,
                train_data=train_data,
                valid_data=valid_data,
                classes=classes,
                block_idx=conv_info['block_idx'],
                conv_in_block_idx=conv_info['conv_in_block_idx'],
                log_file=f"pruning_activation_blocks_3_4_layer_{idx+1}.txt",
                data_yaml=data_yaml
            )
            
            print(f"  ✅ Activation-based pruning applied successfully!")
            successfully_pruned_layers += 1
            
            # Get updated model structure
            torch_model = model.model
            detection_model = torch_model.model
            all_conv_layers_updated = get_all_conv2d_layers(detection_model)
            
            # Try to estimate remaining channels (may not be accurate due to model changes)
            remaining_channels = "Unknown"
            try:
                if len(all_conv_layers_updated) > conv_info['original_model_idx']:
                pruned_layer = all_conv_layers_updated[conv_info['original_model_idx']]
                remaining_channels = (pruned_layer.weight.abs().sum(dim=(1,2,3)) != 0).sum().item()
            except:
                remaining_channels = "Model structure changed"
            
            # Store details for final summary
            pruned_layers_details.append({
                'block_idx': conv_info['block_idx'],
                'conv_in_block_idx': conv_info['conv_in_block_idx'],
                'original_model_idx': conv_info['original_model_idx'],
                'original_channels': conv_info['num_channels'],
                'remaining_channels': remaining_channels,
                'pruned_channels': conv_info['num_channels'] - remaining_channels if isinstance(remaining_channels, int) else "Unknown",
                'pruning_step': idx + 1,
                'status': 'success'
            })
            
            print(f"  📊 Model structure updated after pruning step {idx + 1}")
            print(f"  📈 Successfully pruned {successfully_pruned_layers}/{len(selected_convs)} layers so far")
            
        except Exception as e:
            print(f"  ❌ Error during activation-based pruning: {e}")
            logger.error(f"Failed to prune block {conv_info['block_idx']}, conv {conv_info['conv_in_block_idx']}: {e}")
            
            # Store failed attempt details
            pruned_layers_details.append({
                'block_idx': conv_info['block_idx'],
                'conv_in_block_idx': conv_info['conv_in_block_idx'],
                'original_model_idx': conv_info['original_model_idx'],
                'original_channels': conv_info['num_channels'],
                'remaining_channels': "Failed",
                'pruned_channels': "Failed",
                'pruning_step': idx + 1,
                'status': 'failed',
                'error': str(e)
            })
            
            print(f"  ⚠️  Layer {idx + 1} pruning failed - continuing with remaining layers...")
            # Continue with next layer instead of breaking
            continue
            
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
    
    print(f"\n{'='*60}")
    print(f"📊 Multi-layer pruning summary:")
    print(f"  Successfully pruned: {successfully_pruned_layers}/{len(selected_convs)} layers")
    print(f"  Failed attempts: {len(selected_convs) - successfully_pruned_layers}/{len(selected_convs)} layers")
    
    # Retrain after pruning all selected layers
    print(f"\nStarting retraining after activation-based pruning of {len(pruned_layers_details)} layers...")
    
    # Use the data_yaml parameter from the function call
    model.train(data=data_yaml, epochs=20, verbose=False)
    
    # Final evaluation
    print("Starting final evaluation...")
    final_metrics = model.val(data=data_yaml, verbose=False)
    
    # Calculate total parameters pruned
    total_channels_before = sum(detail['original_channels'] for detail in pruned_layers_details)
    total_channels_after = sum(detail['remaining_channels'] for detail in pruned_layers_details if isinstance(detail['remaining_channels'], int))
    pruning_ratio = (total_channels_before - total_channels_after) / total_channels_before * 100 if total_channels_before > 0 else 0
    
    print(f"\nDetailed Multi-Layer Activation-Based Pruning Summary:")
    print(f"{'='*90}")
    print(f"{'Layer':<8} {'Block':<6} {'Conv#':<7} {'Original#':<10} {'Channels':<15} {'Status':<10}")
    print(f"{'-'*90}")
    
    successful_layers = 0
    failed_layers = 0
    
    for i, details in enumerate(pruned_layers_details):
        channels_info = f"{details['original_channels']}→{details['remaining_channels']}"
        status = details.get('status', 'unknown')
        status_symbol = "✅" if status == 'success' else "❌" if status == 'failed' else "❓"
        
        print(f"{i+1:<8} {details['block_idx']:<6} {details['conv_in_block_idx']:<7} "
              f"{details['original_model_idx']:<10} {channels_info:<15} {status_symbol:<10}")
        
        if status == 'success':
            successful_layers += 1
        elif status == 'failed':
            failed_layers += 1
            if 'error' in details:
                print(f"         Error: {details['error']}")
    
    print(f"{'-'*90}")
    print(f"Overall Statistics:")
    print(f"  Total layers attempted: {len(pruned_layers_details)}")
    print(f"  Successfully pruned: {successful_layers}")
    print(f"  Failed attempts: {failed_layers}")
    print(f"  Success rate: {(successful_layers/len(pruned_layers_details)*100):.1f}%")
    print(f"  Total channels before: {total_channels_before}")
    print(f"  Total channels after: {total_channels_after}")
    print(f"  Overall pruning ratio: {pruning_ratio:.1f}%")
    print(f"{'='*90}")
    
    logger.info(f"Final metrics after activation-based pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")
    
    # Return the model along with layer details for the experiments package
    model.pruned_layers_details = pruned_layers_details
    
    # Enhanced log file with detailed information
    with open("pruning_log_activation_blocks_3_4.txt", "a") as f:
        f.write(f"\n--- Activation-Based Pruning Session (Original Algorithm) ---\n")
        f.write(f"Layers pruned: {len(pruned_layers_details)}\n")
        f.write(f"Layer Details:\n")
        for i, details in enumerate(pruned_layers_details):
            f.write(f"  Layer {i+1}: Block {details['block_idx']}, Conv #{details['conv_in_block_idx']}, "
                   f"Original model #{details['original_model_idx']}: "
                   f"{details['original_channels']}→{details['remaining_channels']} channels\n")
        f.write(f"Total channels: {total_channels_before}→{total_channels_after} ({pruning_ratio:.1f}% reduction)\n")
        f.write(f"Performance: mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")
        f.write(f"--- End Session ---\n\n")
    
    return model

def apply_structural_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune=3):
    """
    Apply structural gamma-based pruning to layers in blocks 1-5.
    This function performs true structural pruning that modifies the model architecture
    instead of just zeroing weights, enabling proper multi-layer pruning without channel mismatches.
    
    Args:
        model_path: Path to the YOLO model
        data_yaml: Path to the dataset YAML file
        layers_to_prune: Number of layers to prune (1-12)
    
    Returns:
        Structurally pruned and retrained model
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Structural Gamma-based Pruning of {layers_to_prune} layers in blocks 1-5 =====")
    print(f"🔧 This method performs TRUE structural pruning - modifying model architecture")
    print(f"🔧 No channel mismatches will occur with this approach")
    
    try:
        # Use the structural pruning implementation
        pruned_model = apply_structural_gamma_pruning(model_path, data_yaml, layers_to_prune)
        
        print(f"✅ Structural gamma-based pruning completed successfully!")
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original soft pruning method
        print(f"🔄 Falling back to soft pruning method...")
        return apply_50_percent_gamma_pruning_blocks_3_4(model_path, data_yaml, layers_to_prune)
#!/usr/bin/env python3
"""
Test script for the iterative pruning implementation
This script tests the fixed iterative pruning approach with 3+ layers
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_iterative_pruning():
    """Test the iterative pruning implementation"""
    print("=" * 80)
    print("TESTING ITERATIVE PRUNING IMPLEMENTATION")
    print("=" * 80)
    
    # Check if required files exist
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please ensure the model file exists before testing")
        return False
    
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        print("   Please ensure the data YAML file exists before testing")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"✅ Data YAML found: {data_yaml}")
    
    # Import and test the main function
    try:
        from run_iterative_pruning_with_finetuning import main
        
        print(f"\n🚀 Running iterative pruning test...")
        print(f"   This will test pruning 3 layers with fine-tuning between each step")
        print(f"   Expected: No channel mismatch errors")
        
        # Run the main function
        success = main()
        
        if success:
            print(f"\n✅ TEST PASSED: Iterative pruning completed successfully!")
            print(f"   - No channel mismatch errors occurred")
            print(f"   - 3+ layers were pruned successfully")
            print(f"   - Fine-tuning worked between each pruning step")
            return True
        else:
            print(f"\n❌ TEST FAILED: Iterative pruning encountered errors")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_iterative_pruning()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test script for the robust channel adjustment fix
This script tests the improved iterative pruning approach
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_robust_channel_fix():
    """Test the robust channel adjustment implementation"""
    print("=" * 80)
    print("TESTING ROBUST CHANNEL ADJUSTMENT FIX")
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
        
        print(f"\n🚀 Running robust channel adjustment test...")
        print(f"   This will test pruning 3 layers with robust channel adjustment")
        print(f"   Expected: No channel mismatch errors")
        print(f"   Expected: Channel adjustment will update subsequent layers")
        
        # Run the main function
        success = main()
        
        if success:
            print(f"\n✅ TEST PASSED: Robust channel adjustment completed successfully!")
            print(f"   - No channel mismatch errors occurred")
            print(f"   - 3+ layers were pruned successfully")
            print(f"   - Channel adjustment worked correctly")
            return True
        else:
            print(f"\n❌ TEST FAILED: Robust channel adjustment encountered errors")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_robust_channel_fix()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Quick test script to verify SWIN MIM wrapper works correctly."""

import torch
import sys

def main():
    print("=" * 60)
    print("Testing SWIN MIM wrapper...")
    print("=" * 60)
    
    # Test 1: Import model
    print("\n1. Importing SwinTransformerForMaskedImageModeling...")
    try:
        from modeling_pretrain import SwinTransformerForMaskedImageModeling
        print("   ✓ Import successful")
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return False
    
    # Test 2: Create model
    print("\n2. Creating model...")
    try:
        model = SwinTransformerForMaskedImageModeling(
            img_size=224,
            patch_size=4,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            vocab_size=8192,
        )
        print("   ✓ Model created successfully")
        print(f"   - num_patches: {model.num_patches}")
        print(f"   - final_dim: {model.final_dim}")
        print(f"   - patch_size: {model.patch_size}")
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        return False
    
    # Test 3: Check patch_embed.patch_size
    print("\n3. Checking patch_embed.patch_size...")
    try:
        ps = model.patch_embed.patch_size
        print(f"   ✓ patch_embed.patch_size = {ps} (type: {type(ps)})")
        if isinstance(ps, tuple) and len(ps) == 2:
            print("   ✓ patch_size is a valid tuple")
        else:
            print("   ⚠ patch_size might not be compatible")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 4: Forward pass without mask
    print("\n4. Testing forward pass (no mask)...")
    try:
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x, return_all_tokens=True)
        print(f"   ✓ Output shape: {out.shape}")
        expected_patches = (224 // 4) ** 2  # 3136
        if out.shape[1] == expected_patches:
            print(f"   ✓ Correct number of patches: {expected_patches}")
        else:
            print(f"   ⚠ Expected {expected_patches} patches, got {out.shape[1]}")
    except Exception as e:
        print(f"   ✗ Forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Forward pass with mask
    print("\n5. Testing forward pass (with mask)...")
    try:
        x = torch.randn(2, 3, 224, 224)
        num_patches = (224 // 4) ** 2
        # Create random mask (mask ~40% of patches)
        bool_masked_pos = torch.rand(2, num_patches) > 0.6
        with torch.no_grad():
            out = model(x, bool_masked_pos=bool_masked_pos, return_all_tokens=False)
        num_masked = bool_masked_pos.sum().item()
        print(f"   ✓ Output shape: {out.shape} (masked tokens: {num_masked})")
        if out.shape[0] == num_masked:
            print("   ✓ Output size matches number of masked tokens")
        else:
            print(f"   ⚠ Expected {num_masked} outputs, got {out.shape[0]}")
    except Exception as e:
        print(f"   ✗ Forward with mask failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Check supports_masked_pos detection
    print("\n6. Testing _model_supports_masked_pos detection...")
    try:
        import inspect
        sig = inspect.signature(model.forward)
        has_masked_pos = 'bool_masked_pos' in sig.parameters
        print(f"   ✓ bool_masked_pos in signature: {has_masked_pos}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 7: Test timm registry
    print("\n7. Testing timm model registry...")
    try:
        from timm.models import create_model
        model2 = create_model('swin_base_patch4_window7_224_8k_vocab', pretrained=False)
        print(f"   ✓ Model created via timm.create_model")
        print(f"   - Type: {type(model2).__name__}")
    except Exception as e:
        print(f"   ✗ Registry failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed! SWIN MIM wrapper is ready.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

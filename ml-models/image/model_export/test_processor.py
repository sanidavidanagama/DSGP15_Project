"""
Simple Test Script
Demonstrates how to use the preprocessor service
Run this to verify everything works before integrating with backend
"""

from preprocessor_service import ChildDrawingPreprocessor
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def test_basic_usage():
    """Test 1: Basic usage with file path"""

    print("TEST 1: Basic Usage (File Path)")

    # Initialize preprocessor (do this ONCE)
    preprocessor = ChildDrawingPreprocessor(verbose=True)

    # Process an image
    image_path = "../data/black bg2.jpeg"  # Replace with your test image
    processed = preprocessor.process(image_path)

    print(f"\n Success! Output shape: {processed.shape}")
    print(f"   Output dtype: {processed.dtype}")
    print(f"   Output range: [{processed.min()}, {processed.max()}]")

    # Save result
    Image.fromarray(processed).save("test_output_1.jpg")
    print("   Saved to: test_output_1.jpg")

    return processed


def test_bytes_input():
    """Test 2: Using bytes input (simulates uploaded file)"""
    print("TEST 2: Bytes Input (Simulates Web Upload)")

    preprocessor = ChildDrawingPreprocessor(verbose=True)

    # Read image as bytes (this is what you get from web uploads)
    image_path = "../data/black bg2.jpeg"
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    print(f"Received {len(image_bytes)} bytes")

    # Process bytes
    processed = preprocessor.process(image_bytes)

    print(f"\nSuccess! Processed {len(image_bytes)} bytes -> {processed.shape} array")

    # Save result
    Image.fromarray(processed).save("test_output_2.jpg")
    print("   Saved to: test_output_2.jpg")

    return processed


def test_pil_input():
    """Test 3: Using PIL Image input"""
    print("TEST 3: PIL Image Input")

    preprocessor = ChildDrawingPreprocessor(verbose=True)

    # Open as PIL Image
    image_path = "../data/black bg2.jpeg"
    pil_img = Image.open(image_path)

    print(f"PIL Image: {pil_img.size}, {pil_img.mode}")

    # Process PIL Image
    processed = preprocessor.process(pil_img)

    print(f"\nSuccess! Processed PIL Image -> {processed.shape} array")

    return processed


def test_numpy_input():
    """Test 4: Using numpy array input"""
    print("TEST 4: NumPy Array Input")

    preprocessor = ChildDrawingPreprocessor(verbose=True)

    # Create numpy array from image
    image_path = "../data/black bg2.jpeg"
    img_array = np.array(Image.open(image_path))

    print(f"NumPy array: {img_array.shape}, {img_array.dtype}")

    # Process numpy array
    processed = preprocessor.process(img_array)

    print(f"\nSuccess! Processed numpy array → {processed.shape} array")

    return processed


def test_process_to_bytes():
    """Test 5: Output as bytes (for API responses)"""
    print("TEST 5: Output as Bytes (API Response)")

    preprocessor = ChildDrawingPreprocessor(verbose=True)

    image_path = "../data/black bg2.jpeg"

    # Process and get bytes
    output_bytes = preprocessor.process_to_bytes(image_path, format='JPEG')

    print(f"\nSuccess! Output: {len(output_bytes)} bytes (JPEG)")

    # Save to verify
    with open("test_output_5.jpg", 'wb') as f:
        f.write(output_bytes)
    print("   Saved to: test_output_5.jpg")

    return output_bytes


def test_different_sizes():
    """Test 6: Custom output sizes"""
    print("TEST 6: Custom Output Sizes")

    preprocessor = ChildDrawingPreprocessor(verbose=True)

    image_path = "../data/black bg2.jpeg"

    # Test different sizes
    sizes = [
        (512, 362),  # Default
        (256, 256),  # Square
        (1024, 724),  # Larger
    ]

    for i, (w, h) in enumerate(sizes):
        processed = preprocessor.process(image_path, target_width=w, target_height=h)
        print(f"   Size {w}×{h}: {processed.shape}")
        Image.fromarray(processed).save(f"test_output_6_{w}x{h}.jpg")

    print("\nSuccess! All sizes processed")



def visualize_pipeline():
    """Test 8: Visualize the pipeline steps"""
    print("TEST 8: Pipeline Visualization")

    preprocessor = ChildDrawingPreprocessor(verbose=True)

    image_path = "../data/black bg2.jpeg"
    result = preprocessor.process(image_path)

    # Display result
    plt.figure(figsize=(10, 7))
    plt.imshow(result)
    plt.title("Final Preprocessed Output", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("test_visualization.png", dpi=150, bbox_inches='tight')
    print("\nSaved visualization to: test_visualization.png")
    plt.show()


def simulate_web_pipeline():
    """Test 9: Simulate complete web pipeline"""
    print("TEST 9: Simulate Complete Web Pipeline")

    # This simulates what happens in production:

    # 1. Server starts - initialize preprocessor ONCE
    preprocessor = ChildDrawingPreprocessor(verbose=False)

    # 2. User uploads image (simulated)
    with open("../data/black bg2.jpeg", 'rb') as f:
        uploaded_bytes = f.read()

    # 3. Preprocess image
    processed_image = preprocessor.process(uploaded_bytes)

    # 4. Pass to mood model (mocked)
    print("🤖 [MOOD MODEL] Analyzing...")
    # mood_prediction = mood_model.predict(processed_image)
    mood_prediction = {
        'mood': 'happy',
        'confidence': 0.85
    }
    print(f"[MOOD MODEL] Prediction: {mood_prediction['mood']} ({mood_prediction['confidence']:.0%})\n")

    # 5. Return to user
    response = {
        'success': True,
        'mood': mood_prediction['mood'],
        'confidence': mood_prediction['confidence']
    }

    return response


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("PREPROCESSOR SERVICE TEST SUITE")

    try:
        # Run tests
        test_basic_usage()
        test_bytes_input()
        test_pil_input()
        test_numpy_input()
        test_process_to_bytes()
        test_different_sizes()
        visualize_pipeline()
        simulate_web_pipeline()


    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback

        traceback.print_exc()
import cv2
import sys
from inference import EdgeAgent

def test_vision_encoder(image_path: str):
    """
    Standalone script to test (i) the Vision Encoder and Processor 
    without any of the (ii) Camera Extraction logic.
    """
    print(f"\n📸 Loading test image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Could not find or load image at '{image_path}'")
        print("Please make sure the file exists.")
        sys.exit(1)

    print("\n🧠 Booting the Vision Encoder (PyTorch)...")
    agent = EdgeAgent()

    # The strict instruction we want to test
    strict_query = "Read the text in the image. Write the factual answer to that question. Do NOT transcribe the question itself."

    print(f"\n📝 Prompting AI with: '{strict_query}'")
    print("\n" + "-" * 50)
    # The AI now yields Agentic tool responses instead of token streams
    stream = agent.generate_stream(image, strict_query)
    for generated_block in stream:
        print(generated_block, end="", flush=True)
    
    print("\n" + "-" * 50)
    print("✅ Test Complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_vision.py <path_to_image>")
        print("Example: python test_vision.py test_image.jpg")
        sys.exit(1)
        
    test_vision_encoder(sys.argv[1])

### BACKUP of the file running on the remote cluster
### TODO: integrate this file at a more appropriate location in the git

# Install environment with:
# conda create -n "molmo" pytorch torchvision torchaudio pytorch-cuda=12.1 numpy pandas matplotlib seaborn ipython ipykernel pylint jupyter pyqt cudatoolkit scipy scikit-learn pip transformers einops accelerate tensorflow pynvml -c pytorch -c nvidia -c conda-forge


# Install instructions on panda1gpu5090 (with RTX5090):
# conda create -n molmo python=3.10 -y
# conda activate molmo
# pip install torch torchvision torchaudio
# pip install transformers==4.45 einops pillow tensorflow accelerate pyzmq


# Run with e.g.:
#   conda activate molmo
#   CUDA_VISIBLE_DEVICES=4; python molmo_zmq_server.py

# or in REPL


# To connect to this, forward port 5555 from the client to the server
# ssh -L 5555:127.0.0.1:5555 <username>@<server-ip>
# ssh -L 5582:127.0.0.1:5582 ott@avalon1


# MISC: improved inference speed by using torch.bfloat16, device_map="cuda" and use_cache=True
# torch compile or torch.inference_mode() did not help here


PORT_NUMBER = 5583


import zmq
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch

# Load the processor and model
processor = AutoProcessor.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Set up ZeroMQ
context = zmq.Context()
socket = context.socket(zmq.REP)  # Reply socket
# socket.bind(f"tcp://*:{PORT_NUMBER}")  # Bind to port 5555
socket.bind(
    f"tcp://127.0.0.1:{PORT_NUMBER}"
)  # Bind to localhost only (setup with ssh port forwarding)

print("MolMo ZMQ Server is running on PORT: ", PORT_NUMBER)

while True:
    # Receive the request
    message = socket.recv_json()
    print("[DEBUG] Received message")

    image_data = base64.b64decode(message["image"])
    text = message["text"]

    # Process the image
    image = Image.open(BytesIO(image_data))

    print(f"[DEBUG] Image dimensions: {image.size} (width x height)")
    print(f"[DEBUG] Text prompt: {text}")

    inputs = processor.process(images=[image], text=text)
    inputs = {
        k: (
            v.to(model.device).to(model.dtype)
            if v.dtype == torch.float32
            else v.to(model.device)
        )
        for k, v in inputs.items()
    }
    inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}

    # Generate output
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(
            max_new_tokens=400, stop_strings="<|endoftext|>", use_cache=True
        ),
        tokenizer=processor.tokenizer,
    )
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )
    print("[DEBUG] Generated text")

    # Send the response
    socket.send_json({"generated_text": generated_text})
    print("[DEBUG] Sent response")

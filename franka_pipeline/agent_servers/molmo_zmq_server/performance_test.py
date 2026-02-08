"""
A performance test script, measuring the time taken for processing and generation, to experiment with different kinds of optimizations.
"""

import time
import torch
from PIL import Image
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def run_performance_test(model_id="allenai/Molmo-7B-D-0924", num_iterations=10):
    print(f"Loading model and processor: {model_id}...")
    start_load = time.time()

    # processor = AutoProcessor.from_pretrained(
    #     model_id,
    #     trust_remote_code=True,
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16,
    #     # device_map={"": "cuda"},
    #     device_map="cuda",
    #     # device_map="auto",
    #     # attn_implementation="flash_attention_2",
    # )

    ### Original
    # # load the processor
    # processor = AutoProcessor.from_pretrained(
    #     "allenai/Molmo-7B-D-0924",
    #     trust_remote_code=True,
    #     torch_dtype="auto",
    #     device_map="auto",
    # )

    # # load the model
    # model = AutoModelForCausalLM.from_pretrained(
    #     "allenai/Molmo-7B-D-0924",
    #     trust_remote_code=True,
    #     torch_dtype="auto",
    #     device_map="auto",
    # )

    ###
    # load the processor
    processor = AutoProcessor.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    print(f"Model loaded in {time.time() - start_load:.2f} seconds.")

    # Optional: torch.compile
    # print("Compiling model (this may take a few minutes)...")
    # start_compile = time.time()
    # model = torch.compile(model)
    # print(f"Model compiled in {time.time() - start_compile:.2f} seconds.")

    # Create a dummy image (RGB, 224x224)
    image = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    text = "Describe this image."

    print(f"Starting performance test with {num_iterations} iterations...")

    latencies = []
    process_times = []
    generate_times = []

    # Warmup
    print("Warming up...")
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
    _ = model.generate_from_batch(
        inputs,
        GenerationConfig(
            max_new_tokens=10, stop_strings="<|endoftext|>", use_cache=True
        ),
        tokenizer=processor.tokenizer,
    )

    for i in range(num_iterations):
        start_iter = time.time()

        # 1. Processing
        start_process = time.time()
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
        process_time = time.time() - start_process

        # 2. Generation
        start_generate = time.time()
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
        generate_time = time.time() - start_generate

        iter_time = time.time() - start_iter

        latencies.append(iter_time)
        process_times.append(process_time)
        generate_times.append(generate_time)

        print(
            f"Iteration {i+1}/{num_iterations}: {iter_time:.4f}s (Process: {process_time:.4f}s, Generate: {generate_time:.4f}s)"
        )

    avg_latency = sum(latencies) / num_iterations
    avg_process = sum(process_times) / num_iterations
    avg_generate = sum(generate_times) / num_iterations

    print("\n--- Results ---")
    print(f"Average Latency: {avg_latency:.4f}s")
    print(f"Average Process Time: {avg_process:.4f}s")
    print(f"Average Generate Time: {avg_generate:.4f}s")
    print(f"Throughput: {1/avg_latency:.2f} iterations/sec")


if __name__ == "__main__":
    run_performance_test()

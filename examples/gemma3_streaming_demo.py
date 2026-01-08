# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streaming sampling demo for Gemma3-1B-IT.

This script demonstrates how to use the streaming sampler to generate
text token-by-token from the Gemma3-1B-IT model.

Usage:
  python examples/gemma3_streaming_demo.py
"""

import sys
import warnings

import jax

from tunix.generate import sampler as sampler_lib
from tunix.models.gemma3 import model as gemma3_model
from tunix.models.gemma3 import params as gemma3_params

# Configuration
MAX_PROMPT_LENGTH = 256
MAX_GENERATION_STEPS = 256
CACHE_SIZE = MAX_PROMPT_LENGTH + MAX_GENERATION_STEPS + 256

# Chat template
TEMPLATE = """<start_of_turn>user
{message}<end_of_turn>
<start_of_turn>model
"""


def main():
  # Setup mesh based on available devices
  # num_devices = len(jax.devices())
  # if num_devices >= 4:
  #   mesh = jax.make_mesh((1, num_devices), ("fsdp", "tp"))
  # else:
  #   mesh = jax.make_mesh((num_devices,), ("fsdp",))
  # print(f"Using {num_devices} device(s)")

  model_config = gemma3_model.ModelConfig.gemma3_1b_it()
  model = gemma3_params.create_model_from_checkpoint(
      gemma3_params.GEMMA3_1B_IT, model_config
  )
  tokenizer = gemma3_params.create_tokenizer()

  # EOS tokens
  eos_tokens = [106]
  if tokenizer.eos_id() not in eos_tokens:
    eos_tokens.append(tokenizer.eos_id())

  # Create sampler
  print("Creating sampler...")
  sampler = sampler_lib.Sampler(
      transformer=model,
      tokenizer=tokenizer,
      cache_config=sampler_lib.CacheConfig(
          cache_size=CACHE_SIZE,
          num_layers=model_config.num_layers,
          num_kv_heads=model_config.num_kv_heads,
          head_dim=model_config.head_dim,
      ),
  )

  # Interactive loop
  print("\n" + "=" * 60)
  print("Ready! Enter your message (or 'quit' to exit)")
  print("=" * 60)

  while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
      print("Goodbye!")
      break

    if not user_input:
      continue

    prompt = TEMPLATE.format(message=user_input)
    print("\nGemma: ", end="", flush=True)

    for output in sampler.stream(
        prompt,
        max_generation_steps=MAX_GENERATION_STEPS,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        eos_tokens=eos_tokens,
    ):
      sys.stdout.write(output.text)
      sys.stdout.flush()

      if output.done:
        break

    print()


if __name__ == "__main__":
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  main()

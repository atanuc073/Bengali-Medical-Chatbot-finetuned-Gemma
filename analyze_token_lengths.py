"""Analyze token lengths of the ChatDoctor-HealthCareMagic-100k dataset.
Uses a simple word-based approximation (words * 1.3 ≈ subword tokens for English).
"""

from datasets import load_dataset
import statistics

print("Loading dataset...")
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
print(f"Total samples: {len(dataset)}\n")

def approx_tokens(text):
    """Approximate subword token count: ~1.3 tokens per whitespace word for English."""
    if not text or not text.strip():
        return 0
    return int(len(text.split()) * 1.3)

instruction_lens = []
input_lens = []
output_lens = []

for sample in dataset:
    instruction_lens.append(approx_tokens(sample["instruction"]))
    input_lens.append(approx_tokens(sample["input"]))
    output_lens.append(approx_tokens(sample["output"]))

def print_stats(name, lengths):
    lengths_sorted = sorted(lengths)
    n = len(lengths)
    print(f"{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Min:             {min(lengths)}")
    print(f"  Max:             {max(lengths)}")
    print(f"  Mean:            {statistics.mean(lengths):.1f}")
    print(f"  Median:          {statistics.median(lengths):.1f}")
    print(f"  Std Dev:         {statistics.stdev(lengths):.1f}")
    print(f"  P75:             {lengths_sorted[int(n * 0.75)]}")
    print(f"  P90:             {lengths_sorted[int(n * 0.90)]}")
    print(f"  P95:             {lengths_sorted[int(n * 0.95)]}")
    print(f"  P99:             {lengths_sorted[int(n * 0.99)]}")
    print(f"  > 128 tokens:    {sum(1 for l in lengths if l > 128):>6} ({sum(1 for l in lengths if l > 128)/n*100:.1f}%)")
    print(f"  > 256 tokens:    {sum(1 for l in lengths if l > 256):>6} ({sum(1 for l in lengths if l > 256)/n*100:.1f}%)")
    print(f"  > 512 tokens:    {sum(1 for l in lengths if l > 512):>6} ({sum(1 for l in lengths if l > 512)/n*100:.1f}%)")
    print()

print_stats("INSTRUCTION", instruction_lens)
print_stats("INPUT (patient question)", input_lens)
print_stats("OUTPUT (doctor response)", output_lens)

combined = [i + o for i, o in zip(input_lens, output_lens)]
print_stats("INPUT + OUTPUT combined", combined)

# Unique instructions
unique_instructions = set(dataset["instruction"])
print(f"\nUnique instructions: {len(unique_instructions)}")
for inst in list(unique_instructions)[:5]:
    print(f"  • \"{inst[:100]}\"")

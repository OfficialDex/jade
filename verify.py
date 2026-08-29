import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import dataset as dataset_configs
from configs import model as model_configs
from core import data, tokenize, train


def main():
    raw_dir = Path("data/raw")
    if not raw_dir.exists() or not any(raw_dir.glob("shard-*.jsonl")):
        print("no dataset shards found, building a small sample from fineweb + common pile")
        manifest_data = data.build(dataset_configs.tiny, raw_dir)
        print(f"built {manifest_data['deduped_count']} records across {len(manifest_data['shards'])} shards")
    else:
        print("using existing dataset shards in data/raw")

    vocab_size = model_configs.tiny["vocab_size"]
    print(f"training jade tokenizer, vocab_size={vocab_size}")
    output_path = tokenize.train(raw_dir, vocab_size=vocab_size)
    print(f"saved to {output_path}")

    sample = "Jade is a research project for efficient language models."

    tokenize.load(output_path)
    assert tokenize._mode == "jade", f"expected mode 'jade', got {tokenize._mode}"
    ids_first = tokenize.encode(sample)
    decoded_first = tokenize.decode(ids_first)
    print(f"encode/decode round-trip: {decoded_first == sample}")

    tokenize._tokenizer = None
    tokenize._mode = None
    tokenize.load(output_path)
    ids_second = tokenize.encode(sample)
    assert ids_first == ids_second, "reloaded tokenizer produced different ids than the freshly trained one"
    print("independent save/load verified: reloaded tokenizer matches trained tokenizer exactly")

    print(f"vocab_size from disk: {tokenize.vocab_size()}")

    token_dir = Path("data/tokens")
    token_manifest = tokenize.tokenize_shards(raw_dir, token_dir)
    print(f"tokenized {token_manifest['total_tokens']} tokens across {len(token_manifest['shards'])} shards")

    shards = train.load_tokens(token_dir)
    total = sum(len(s) for s in shards)
    assert total == token_manifest["total_tokens"], "train.py's loader disagrees with the tokenize manifest"
    print(f"core/train.py can read the shard directory: {total} tokens across {len(shards)} files")

    stats = tokenize.benchmark(sample, iterations=200)
    print(f"benchmark: {stats['tokens_per_sec']:.0f} tokens/sec, {stats['compression_ratio']:.2f} chars per token")

    print("m4 verification complete")


if __name__ == "__main__":
    main()


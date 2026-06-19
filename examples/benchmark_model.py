"""Example: Benchmark all model variants."""

from flashcls.analytics import Benchmark


def main():
    bench = Benchmark(device="cuda")
    print("FlashCls Benchmark")
    print("=" * 60)
    results = bench.compare_all(input_size=224, num_classes=1000)
    for r in results:
        print(f"  {r['model_size']:<10} {r['params']:>8,} params  {r['latency_ms']:>6.2f} ms  {r['throughput_fps']:>6.0f} FPS")


if __name__ == "__main__":
    main()

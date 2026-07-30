# Run SLAKE only, 3 seeds
cmd = argparse.Namespace(
    benchmark="slake",
    checkpoint="/content/drive/MyDrive/qfsru_checkpoints/pretrain_epoch_20.pt"
)

import statistics as stats
all_results = []
for seed in (42, 123, 2024):
    print(f"\n{'='*80}\n🌱 SLAKE run with seed={seed}\n{'='*80}")
    result = run_single_seed(cmd, seed)
    all_results.append(result)

ems = [r["overall_em"] for r in all_results if r["overall_em"] is not None]
print(f"\n{'='*80}\n📊 SLAKE MULTI-SEED SUMMARY (3 seeds)")
for r in all_results:
    print(f"  seed={r['seed']:<6} overall_em={r['overall_em']}")
if ems:
    print(f"  MEAN EM: {stats.mean(ems):.4f}  ±  STD: {stats.stdev(ems):.4f}")
print(f"{'='*80}")

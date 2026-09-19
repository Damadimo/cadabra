# Paper

`cadabra.tex` — a short write-up of the task, data, training, grading and results.

```sh
brew install tectonic      # once
tectonic cadabra.tex    # -> cadabra.pdf
```

Figures in `figures/` are copies of `docs/*.png` plus one benchmark drawing sheet. Rebuild them and the numbers with:

```sh
python scripts/leaderboard.py --latest sota-img,sota-img-rest,ours-img,ours-img-bo8,ckpt200-img,base-img-rest \
  --common --out-json data/demo/scoreboard.json     # the table in Section 6
python scripts/plot_results.py                      # results_by_tier.png, cost_vs_accuracy.png
python scripts/plot_curve.py --final-tag ours-img   # learning_curve.png
python scripts/leakage_check.py --ours ours-img-bo8 # the near-duplicate table in Section 7
cp docs/*.png paper/figures/
```

Current version reports the 1-epoch model (training job `wgpp6vw`). When a later run wins on the same parts, rerun the
commands above with its tag and update: the abstract, Table 2, Table 3, Section 6 prose, and the training time and step
count in Section 3.

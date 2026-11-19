# NeuroSync FYP - Evidence Log (Saad Arif, SRN 25062380, 6WCM0029)
Purpose: every piece of work that could be "shown in or referenced from" the IPR/FPR (IPR rule: unreferenced work cannot normally be credited).
Add one row per item. Never record a result that was not actually observed. AI help must be declared in the Interim/Final AI Use Statements.

| ID | Date/time (UK/Karachi) | What was done | Evidence (file / link) | Source of truth | AI-assisted? | Use in IPR/FPR |
|----|------------------------|---------------|------------------------|-----------------|--------------|----------------|
| E01 | 27-Oct | Repo review of existing NeuroSync (GitHub): EEGEncoder hardcoded 64x1000, random-data loader, hardcoded API confidences, shape-only tests, CI masks failures | Repo on Saad's GitHub; review in chat | Repo code | Review by Claude | IPR: starting point / honest baseline of prior work (confirm with supervisor if pre-module work counts) |
| E02 | 2-Nov | Located dataset THINGS-EEG2 on OSF (project 3jk45) and verified components via OSF API | URLs in E03 | OSF | Searched by Claude | IPR: dataset choice/justification; FPR: data section |
| E03 | 2-Nov 16:11 | Verified files: sub-01.zip 1,071,722,467 B (osf.io/download/nb8wr/); test_images.zip 8,129,948 B (znu7b); training_images.zip 655,039,265 B (3v527); image_metadata.npy 656,579 B (qkgtf). Re-checked in Colab with os.path.getsize: all four byte-exact | E03_E04_step1_rerun.jpg | OSF | Run by Claude via browser | Provenance |
| E04 | 2-Nov 15:40, re-run 16:11 | Step 1 data check: train (16540,4,17,100); test (200,80,17,100); 17 channels (Pz, P3, P7, O1, Oz...); -0.2..0.79 s, 100 samples; 0 NaN; mean -0.035; std 0.8401. Re-run with timestamp printed for evidence | E03_E04_step1_rerun.jpg; colab_baseline_full.py cell 2 | Colab output | Code by Claude | IPR criterion 4 (technical progress) |
| E05 | 2-Nov ~15:55 | T4 GPU allocated (Tesla T4, 15360 MiB); images + metadata downloaded; metadata: 16540 train / 200 test files, order matches EEG index | screenshots | Colab output | Claude | Reproducibility |
| E06 | 2-Nov ~16:03 | CLIP ViT-B/32 (LAION-2B) features extracted: all 16740 images matched; train (16540,512), test (200,512) | screenshots; cell 5 | Colab output | Claude | Method |
| E07 | 2-Nov ~16:05 | Baseline trained (40 epochs, seed 0). Real: top-1 30.0%, top-5 ~61%. Shuffled-label control: 0.5% / ~3%. Chance 0.5% / 2.5%. NICE paper abstract (arXiv 2308.13234v3) reports 15.6% / 42.8% (setups differ, not directly comparable) | screenshots; cell 6 | Colab output | Claude | IPR criterion 4/evaluation; FPR results |
| E08 | 2-Nov ~16:17 | NeuroSync repo adapted to real data: new eeg/things_dataset.py, eeg/baseline_encoder.py, eeg/metrics.py, clip_loss (learnable temperature) added to models/loss.py, experiments/train_things_eeg.py, tests/test_things_eeg.py (6 tests: shapes, retrieval metric, loss, repetition averaging, shuffle control). Original EEGEncoder left unchanged and confirmed to accept 17x100 input. Tests run on Colab: 6 passed | E08_repo_tests_6_passed.jpg; repo files listed | Colab output | Code by Claude; Saad to review/understand | IPR criterion 4 (technical progress, testing) |
| E09 | 2-Nov ~16:22 | Repo-code results, subject 1, 40 epochs, final epoch, 200-way retrieval. ConvEEGBaseline seeds 0/1/2: top-1 23.5 / 30.0 / 26.0 %, top-5 52.5 / 60.0 / 52.0 %. Shuffled-label control (seed 0): 1.0 / 5.0 %. Original NeuroSync CNN+Transformer EEGEncoder (seed 0): 24.5 / 49.5 %. Chance 0.5 / 2.5 % | E09_repo_seed_sweep.jpg | Colab output | Run by Claude | IPR/FPR results: seed variance (~6 pts top-1) must be reported, not a single best run |
| E10 | 19-Nov 17:27-18:19 Karachi (12:27-13:19 UTC) | Multi-subject run, THINGS-EEG2 subjects 1-10: per subject 3 seeds (0,1,2) of ConvEEGBaseline-equivalent encoder + 1 shuffled-label control (seed 0); 40 epochs, final epoch only, 200-way retrieval vs CLIP ViT-B/32 (LAION-2B) features. Per-subject mean top-1/top-5 (%): S01 25.7/56.2, S02 23.8/56.2, S03 28.0/63.7, S04 27.8/63.3, S05 20.5/46.8, S06 36.5/66.3, S07 31.5/61.0, S08 39.2/72.3, S09 30.7/58.5, S10 41.5/74.2. Across subjects: top-1 30.5 +/- 6.8, top-5 61.9 +/- 8.1 (SD over 10 subject means). Shuffled controls 0.0-0.5% top-1, 1.5-4.5% top-5 (chance 0.5/2.5). Subject zips size-checked against OSF byte counts | E10_multisubject_summary.jpg; experiments/colab_multisubject.py | Colab output (T4) | Code by Claude; Saad to review/understand | FPR results: generalisation across subjects, between-subject variability |
## Caveats to state honestly in the report
- One subject, no hyperparameter search, final epoch reported (no test-set epoch selection). E07 (notebook, seed 0 stream) gave 30.0%; repo code over 3 seeds gave 23.5-30.0% top-1, so quote the range/mean (26.5% top-1, 54.8% top-5), not the best run.
- Shuffled control at 1.0%/5.0% is 2/200 and 10/200 correct: consistent with chance on a 200-item test set (noise is about +/-0.7 pts on top-1).
- Original transformer encoder (24.5% / 49.5%) is within the baseline's seed range: the existing NeuroSync architecture is viable.
- Higher than the paper: differences in setup (80-rep test averaging, LAION CLIP backbone, small encoder) - not a claimed improvement.
- THINGS-EEG2 block-design confound: needs fair baselines/discussion.
- Colab outputs are not saved: screenshots are the only record until the notebook is re-run.

- E10: each subject is trained and tested separately (within-subject), not cross-subject. Seeds here (0,1,2 via torch.manual_seed per run) differ from the earlier repo seed sweep (E09), so subject 1 reads 25.7% mean here vs 26.5% in E09: same setup, different random draws, which itself shows the run-to-run noise (~2-5 pts per subject).
- E10: numbers were read from the Colab output screenshot; the notebook outputs are not saved. Per-subject means cross-check against the printed across-subject means (30.5 / 61.9).
- E10: the pasted cell's header comment differs slightly from experiments/colab_multisubject.py (comment lines only); code is identical.

## To do for evidence
- [x] Colab notebook renamed NeuroSync_Baseline_Step1-2_2Nov2026.ipynb (in Saad's Drive).
- [x] Step 1 re-run with screenshot (E03/E04).
- [ ] Optional: File > Download > .ipynb in Colab and drop it in this folder (code only; outputs are private/not saved, screenshots are the output evidence).
- [ ] Record what Steve Bennett says on Thu 5 Nov (pre-module work, ethics, referencing style).
- [ ] Draft Interim AI Use Statement from the AI-assisted column.

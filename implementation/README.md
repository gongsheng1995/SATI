# Implementation snapshot

Run from this directory after obtaining official benchmark inputs/features and
the pinned BERT checkpoint. Copy the needed frozen cache paths from
`../classifier_inputs/data/`; consult per-run configs under `../runs/`.

The generic SATI entry point is `scripts/train.py`; for example use `--help`
to inspect input paths and training arguments before configuring a separate output
directory. Do not rerun into an existing experiment directory. Dataset label order,
fixed macro universe and ordinary best-dev-accuracy selection must be retained.

PyTorch 2.8.0 must be installed with the CUDA 12.8 build for the documented GPU
environment; the requirements file records version constraints, not a GPU provisioner.
No training is started by packaging this snapshot.

The archived teacher driver is an original execution-bound script and requires
its frozen plan/authorization/model assets to execute. It is included to inspect
the performed computation; the supplied messages/raw responses suffice to audit
the frozen prior without executing it or provisioning a server.

Baseline adapters, original/completion driver snapshots and the small pinned
vendor source files for the five reported architecture baselines are included.
Recorded revisions and upstream licenses apply. Execution guards in original
drivers preserve the historical protocol; supplying new execution bindings is
separate from score verification. No model/feature weights are included.

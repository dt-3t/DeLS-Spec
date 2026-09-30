# DeLS-Spec release checklist

Included in the paired source update:

- Training documentation now lives in the `dt-3t/SpecForge` checkout and points
  to its `add-dels-spec` branch and the DeLS-Spec runtime repository.
- The Qwen3-8B recipe uses the same data, tokenizer, template, and length limit
  for the RNN head and integrated rank-0 prior statistics.
- The updated launcher uses the current CLI, explicit architecture/export
  settings, configurable seed/epochs/batch, and one GPU by default.
- `dels-plots` is an optional matplotlib extra; ordinary prior export does not
  require plotting or a separate statistics launch.
- The accompanying runtime detects fused RNN exports correctly and uses their
  learned/projected embedding without modifying the target embedding. Legacy
  released RNN checkpoints remain supported.
- 22 CPU tests passed, including training/data tests and nine runtime integration
  cases. Shell syntax and mocked launcher/argument checks passed. These checks
  do not run GPU training or benchmark decoding.

Before a full paper-code release:

- Choose the runtime license and confirm authorization for inherited Domino
  code; the inspected source has no LICENSE file. Preserve SpecForge's existing
  MIT license and third-party copyright notices.
- Record the exact training `add-dels-spec` and runtime `main` commits used for
  a reproduction run. Use paired versions containing the current guide.
- Upload the regenerated source archives and SHA256SUMS as release assets if
  distributing snapshots. Each package includes SOURCE_MANIFEST.json.
- Validate a fresh installation with the dependency versions in pyproject.toml;
  CPU tests used an existing older environment, not those complete pins.
- Run a small GPU execution check, then the paired benchmark. Fill the guide's
  result table with actual logs, GPU/backend details, data/model revisions,
  checkpoint selection, and repeated timings for performance claims.
- Add the paper's authoritative citation metadata when available. Model and
  dataset licenses remain separate from code licensing.

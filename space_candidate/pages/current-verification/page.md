# Current verification and visibility matrix

The current verifier supersedes the historical pages for all scientific
decisions. All six claims now have evaluator-visible source contracts,
executable checks, data, controls, and explicit limitations. Claim 6 is
**VERIFIED**. Claims 1–5 remain **BLOCKED** at theorem level; no finite check is
promoted into a universal proof.

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [Claim 1](#/claim-1-current) | [verifier](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/code/verify_current.py) | yes | [JSON](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/data/claim1_full_network.json) | bit-exact rows + identity matrix | rotated gradients | yes | BLOCKED |
| 2 | [Claim 2](#/claim-2-current) | [verifier](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/code/verify_current.py) | yes | [JSON](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/data/claim2_activations.json) | analytic/autograd | constant activation | yes | BLOCKED |
| 3 | [Claim 3](#/claim-3-current) | [verifier](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/code/verify_current.py) | yes | [JSON](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/data/claim3_theorem32.json) | exact-depth + exact rows | even target | yes | BLOCKED |
| 4 | [Claim 4](#/claim-4-current) | [verifier](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/code/verify_current.py) | yes | [JSON](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/data/claim4_zero_gradient.json) | paper-order interpreter | zeroed branch weight | yes | BLOCKED |
| 5 | [Claim 5](#/claim-5-current) | [verifier](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/code/verify_current.py) | yes | [JSON](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/data/claim5_zero_output_gradient.json) | calibrated exact rows | released normalization | yes | BLOCKED |
| 6 | [Claim 6](#/claim-6-current) | [verifier](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/code/verify_current.py) | yes | [JSON](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/data/claim6_raw.json) | exact rational | powers of two | yes | VERIFIED |

## Reproduction command and environment

```bash
uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py
python space_candidate/code/verify_current.py
```

The experiment command is fixed across every node. Python 3.12.12, NumPy
2.3.1, and Torch 2.7.1+cpu are locked by `uv.lock`. Formal runs use Hugging
Face `cpu-upgrade`, never a GPU. Each route estimated one required core; 64
logical/affinity CPUs were visible and Torch was capped at 4 threads. Seeds,
run IDs, Git SHAs, job runtime, and scientific runtime appear on every claim
page and in every linked JSON.

Current cumulative verifier output:

```text
claim=1 evidence_check=PASS theorem_verdict=BLOCKED
claim=2 evidence_check=PASS theorem_verdict=BLOCKED
claim=3 evidence_check=PASS theorem_verdict=BLOCKED
claim=4 evidence_check=PASS theorem_verdict=BLOCKED
claim=5 evidence_check=PASS theorem_verdict=BLOCKED
claim=6 evidence_check=PASS theorem_verdict=VERIFIED
cumulative_regression=PASS controls=PASS
```

Formal cumulative regeneration run
`4064ba4a-a790-48ab-ab16-e7c674c7151b` at Git
`f09a1bcbcdfac1c0ffac71deb792ed8c1e6d13bb` completed on HF
`cpu-upgrade` in 58 s (21.296464 s scientific runtime). It regenerated every
route, checked the displayed JSON against those live results, and produced the
output above.

The release-winning repeat
`b348d98a-3474-4a1a-8fee-8dcd084e0b85` at Git
`e412c472eb5adc6beb00bb311ce7f1fbf698170d` completed on the same CPU flavor
in 64 s (24.209091 s scientific runtime). It corrected stale generated
evaluation wording without changing a scientific route and reran the complete
suite successfully.

The old `verify` and `overview` pages are preserved as **Historical rejected
baseline** evidence and are not current verification.

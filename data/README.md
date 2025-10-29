# InteractComp (Encrypted Release)

InteractComp accompanies the paper **"INTERACTCOMP: Evaluating Search Agents with Ambiguous Queries"** (Deng et al., 2025). The benchmark targets a capability gap in modern search agents: recognizing ambiguity, asking clarifying questions, and only then executing retrieval or answering. Despite rapid progress on fully specified web queries, the paper shows that interaction-centric performance remains stagnant—highlighting InteractComp as both an evaluation suite and a training ground for interactive behaviors.

## What's Inside
- **210 human-authored tasks** spanning 9 web domains. Each instance includes an intentionally ambiguous question, targeted context for disambiguation, the ground-truth answer, and metadata (`id`, `domain`).
- **Target-distractor methodology:** curators select a target entity and a popular distractor, write questions using only shared attributes, then provide minimal context that exposes the target's unique traits. Ambiguity can only be resolved through interaction.
- **Interaction sensitivity:** the paper reports that GPT-5 tops out at **13.73% accuracy** under standard agent settings but exceeds 24% when forced to interact extensively. Supplying the full disambiguating context pushes accuracy to 71.5%, demonstrating that the limiting factor is proactive interaction rather than reasoning.
- **Longitudinal findings:** over 15 months (May 2024 to Sep 2025) agent accuracy on BrowseComp increases seven-fold, yet InteractComp accuracy stagnates between 6% and 14%, exposing a systemic blind spot in agent development.
- **Transparent pipeline:** the appendix details a five-step construction process (Target and Distractor selection -> Shared and Distinct attributes -> Ambiguous question -> Context snippets -> Reasoning path) ensuring each item is challenging to shortcut yet straightforward to verify post-interaction.

### Reference
Please cite the paper when using this release:

> **Mingyi Deng, Lijun Huang, Yani Fan, Jiayi Zhang, et al. (2025)**  
> *INTERACTCOMP: Evaluating Search Agents with Ambiguous Queries*.

## Repository Layout

```
data/
|-- dataset/
|   `-- InteractComp210.encrypted.jsonl   # Encrypted benchmark samples 
|-- interactcomp_decrypt_jsonl.py     # Decryption utility
```

## Encryption Scheme
- A shared canary token (default `InteractComp`) is embedded in the `canary` column of every record.
- Fields `domain`, `question`, `answer`, and `context` are XOR-encrypted byte-wise using a SHA-256 stream derived from that token, then Base64 encoded.

## Decryption Instructions

1. Prepare Python 3.8+.
2. Place the encrypted JSONL and the decryption script in your working directory (or provide explicit paths).
3. Run:

```bash
python interactcomp_decrypt_jsonl.py \
  --input InteractComp210.encrypted.jsonl \
  --output InteractComp210.decrypted.jsonl
```

Optional flags:
- `--fields` to target a different comma-separated list of encrypted columns.
- `--canary-field` if the canary column name differs from `canary`.
- `--canary-value` to supply the token manually when the canary column has been removed from the encrypted file.
- `--keep-canary` to retain the canary column in the decrypted output.

### Verifying the Result
Decrypted records should be valid UTF-8 JSON lines matching the original dataset. Use diff tools or hash comparisons against an authorized plaintext copy. If a record fails to decrypt, confirm that the canary value you used matches the one embedded during encryption.

## Token Management
- Keep the shared canary token confidential if you want to avoid casual decryption. Rotate it and regenerate the encrypted release if exposure is suspected.
- Consider producing per-recipient encrypted variants (unique tokens) when auditing access or tracing potential leaks.

## License & Intended Use
This release is intended for non-commercial research on interactive search agents. Any downstream use should comply with the policies outlined in the accompanying paper and institutional review requirements. Contact the maintainers for questions about redistribution or additional licensing terms.

# Paper source audit

- Paper: arXiv `2605.01702`
- Retrieved URL: `https://ar5iv.labs.arxiv.org/html/2605.01702`
- Retrieval time (UTC): `2026-07-27T08:37:14Z`
- Request User-Agent: `Mozilla/5.0 (compatible; OpenResearch-Repro/1.0; +https://openresearch.ai)`
- Retrieved HTML bytes: `1,621,047`
- SHA-256: `5dee110336720fa632917d6f97c9cb2ad09c9cda2ce809bd2640d64b1fc4d55d`

## Exact anchors and quantifiers

- Assumption 1: `#Thmassumption1.1.1.1`. The paper assumes
  \(2 \le p \le 2^{q-2}\) and \(q \ge 4\).
- Theorem 3.1: `#S3.Thmtheorem1.1.1.1`. For \(q\ge6\), each listed
  activation, the complete bounded floating-point domain
  \(X=[-M_\sigma,M_\sigma]^d_F\), arbitrary \(f^*:X\to F\), bounded
  input-gradient map \(h^*\), and arbitrary \(g^*\) satisfying
  \(h^*(x)=0\Rightarrow g^*(x)=0\), every \(L\ge9\) admits an
  \(L\)-layer network with exact target values and AD gradients for all
  \(x\in X\).
- Theorem 3.2: `#S3.Thmtheorem2.1.1.1`. Under the same format and domain,
  arbitrary \(f^*\) and arbitrary \(g^*(x,y)\) satisfying
  \(g^*(x,-y)=-g^*(x,y)\) are represented by one network for every
  \(x\in X\), every bounded \(y\), and every
  \(L\ge2^{q+1}+2p+11\).
- Lemma 3.4: `#S3.Thmtheorem4.1.1.1`. Under Conditions 1–2 and
  distinguishability, the constructed network represents arbitrary
  values, has exactly zero AD gradient for every stated input gradient,
  and is neutral under the paper's `boxplus` composition.
- Lemma 3.5: `#S3.Thmtheorem5.1.1.1`. Under Conditions 1–3 and
  distinguishability, the constructed network is exactly zero on the
  complete domain, realizes the arbitrary target AD gradient, and is
  neutral under `boxplus`.

These are universal finite-domain statements. A finite random sweep is only
corroboration; it is not treated as a proof of the quantifiers.

## Provenance snapshots

- Authors' code:
  `https://github.com/yechanp/fp-grad-rep@3cf61240748f09af29084556b1876eddc1e462fb`
- Authors' `src/models.py` SHA-256:
  `beccdd9c9ded1e17eaf829f85f17c48229eb6e39c49fb6547728b76414027139`
- Live verdict dataset:
  `https://huggingface.co/datasets/ICML-2026-agent-repro/verdicts/resolve/main/verdicts.json`
- Verdict dataset retrieval date: `2026-07-27`
- Verdict dataset SHA-256:
  `44309d51b7be5328bbdb8d36aaa809c05704d26b88d201b3860897a9acbed63d`
- Filter used: exact equality
  `space_id == "DineshAI/g89qqA6qmD"`.

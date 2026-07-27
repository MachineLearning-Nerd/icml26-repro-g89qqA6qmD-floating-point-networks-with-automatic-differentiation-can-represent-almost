# Judged-to-candidate subset check

Judged revision:
`DineshAI/g89qqA6qmD@56de71e80b8e71a2928cdad2323cc7d7ad877d78`

Protected manifest: `release/judged_revision_manifest.sha256` (`15` paths).

Planned text upload: `release/space_upload_allowlist.txt` (`28` paths), with
hashes in `release/space_upload_manifest.sha256`.

## Path result

| Judged path class | Count | Candidate action | Result |
| --- | ---: | --- | --- |
| text/app paths also in upload | 11 | update or re-upload | retained |
| binary logos and bucket icon | 4 | omitted from text-only upload | retained remotely, untouched |
| total judged paths | 15 | additive commit; no delete operations | `15/15` retained |

The path set of the judged revision is therefore a subset of the planned
published tree. In addition:

- `pages/overview/page.md` retains SHA-256
  `e746aa26210c38a982d76b8ac6aa66b04ad26945396c80c5c7fdad794afc33ae`;
- `pages/verify/page.md` retains SHA-256
  `0cdae12e9ae9c83c5e5417564ab204af40849a2e6fa5995673cae9464abbef40`;
- the original index content retains SHA-256
  `b100c7bd18d2f89b12a4622ebd643667249a71268e7bd6694f5a7109e6ca727c`
  at `pages/historical-index/page.md`;
- the exact judged revision remains immutable and addressable in Hugging Face
  repository history.

The upload implementation must issue only additive/update operations for the
28 allowlisted text paths and no delete operation.

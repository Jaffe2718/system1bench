# External dataset attribution

The repository's MIT license covers the new collection code, not all source data.
Raw question text is built under ignored `data/` and is not committed to GitHub.
The local/Colab bank and result ZIP are research artifacts, not a blanket
commercial-use or redistribution license. Preserve source-specific terms and
attributions when using or sharing them.

Exact download URLs, immutable Hub revisions (where available), source file SHA256
and stated license labels are in `sources.json`. The SocialIQA author-hosted
archive is byte-pinned and contains `LICENCE-CC-BY-4.txt`. Its dev records include
the authoritative `correct` letter; unlabelled test records are not used.

| Source | Attribution / declared terms |
|---|---|
| BoolQ | Clark et al. (2019); source card declares CC-BY-SA-3.0 |
| COPA | Roemmele, Bejan and Gordon (2011); task-specific terms, distributed through SuperGLUE |
| XNLI | Conneau et al. (2018); consult original data terms; Hub metadata does not state a license |
| SocialIQA | Sap et al. (2019); author archive CC-BY-4.0 |
| AG News | Zhang et al. and original AG corpus; Hub license is unknown; consult original terms |
| ARC | Clark et al. (2018), AllenAI; source card declares CC-BY-SA-4.0 |
| SST5 | Socher et al. (2013), Stanford Sentiment Treebank; consult original terms |
| CommonsenseQA | Talmor et al. (2019); tau Hub card declares MIT; retain original attribution |
| Emotion | Saravia et al. (2018), dair-ai; source card uses other/custom terms |
| Banking77 | Casanueva et al. (2020); source card declares MIT |
| MultiRC | Khashabi et al. (2018); original research/academic terms, distributed through SuperGLUE |

SuperGLUE's card says to consult each underlying dataset's original license; it
does not grant a single MIT license to COPA and MultiRC. Dataset availability is
not evidence that it was absent from model training. The builder does not claim
independent human verification of every reference label.

Primary references:
- https://maartensap.com/social-iqa/
- https://cogcomp.seas.upenn.edu/multirc/
- https://huggingface.co/datasets/aps/super_glue
- https://huggingface.co/datasets/allenai/ai2_arc
- https://huggingface.co/datasets/tau/commonsense_qa

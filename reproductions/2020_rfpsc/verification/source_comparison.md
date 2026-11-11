# RFPSC source-file consistency check

**Conclusion: the article content is identical; the repository cover and PDF metadata differ.** No network access was used.

Primary file SHA-256: `8420166351fbbc3b05a2932b55af0f3c17f0edfde8bb556c1ee5b773e9d4d391`.
Independent file SHA-256: `55d87c408fe93ba20b1df49110cf50fcef934cd0ebd962c2667ea9f033bf7f38`.

Both PDFs have seven pages and identify DOI `10.1109/TPEL.2020.2970991`. PDF page 2 (copyright notice) and pages 3-7 (the complete article) have identical extracted text and decoded page drawing streams. All six pages also render to exactly identical RGB pixel buffers at 144 dpi using the same PyMuPDF renderer.

| Item | PDF page | Printed article page | Text / decoded stream / rendered pixels |
|---|---:|---:|---|
| equation_19 | 4 | 2 | identical / identical / identical |
| table_I | 5 | 3 | identical / identical / identical |
| figure_3 | 5 | 3 | identical / identical / identical |
| figure_5 | 6 | 4 | identical / identical / identical |
| figure_6 | 6 | 4 | identical / identical / identical |

The shared article-body extracted-text SHA-256 is:
`68c756dc4ec58fb7b957704af5ecb39c59d7915e4111ce5f43cbde259693c43f`

The shared length-prefixed decoded article-page stream SHA-256 is:
`2947c4edd78fda97f89be4429e16498458820c3fb66cf7f5564237824830e937`

Only the first repository cover differs visibly. The primary cover uses an early-online citation and the label "Peer reviewed version". The independent cover expands the accepted-manuscript description and adds the final volume, page range, and article number. Both retain the same DOI and date.

PDF producer/creation metadata also differ: Adobe Acrobat/2020 versus OpenPDF/2026. A later PDF creation timestamp must not be interpreted as a revised scientific manuscript.

Equation (19), Table I, and Figures 3, 5, and 6 are the same in both copies. In particular, the Ra=4.4 ohm, Zbase=12.6 ohm, Ra=.2 pu inconsistency is present in both. Source-file differences therefore do not explain or resolve that ambiguity.

Keep both original file hashes for provenance. Scientific body equivalence is supported by exact text, decoded-content, and rendered-pixel comparisons; the files themselves remain byte-distinct. Detailed hashes and the cover-text diff are in source_comparison.json.

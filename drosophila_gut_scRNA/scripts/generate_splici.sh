
#!/usr/bin/env bash
set -euo pipefail

GENOME_FASTA="$1"
GTF="$2"
OUT_FASTA="$3"
OUT_T2G="$4"
READ_LENGTH="${5:-91}"  # Default 91bp for 10x Chromium
FLANK_TRIM="${6:-5}"

# Create temporary output directory
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Run pyroe make-splici
echo "Running pyroe make-splici with genome: $GENOME_FASTA, GTF: $GTF, read-length: $READ_LENGTH"
pyroe make-splici \
    "$GENOME_FASTA" \
    "$GTF" \
    "$READ_LENGTH" \
    "$TMPDIR" \
    --flank-trim-length "$FLANK_TRIM" \
    --dedup-seqs

# Move outputs to expected locations
mv "$TMPDIR/splici_fl"*".fa" "$OUT_FASTA"
mv "$TMPDIR/splici_fl"*"_t2g"*".tsv" "$OUT_T2G"

echo "Successfully generated splici reference and t2g mapping"

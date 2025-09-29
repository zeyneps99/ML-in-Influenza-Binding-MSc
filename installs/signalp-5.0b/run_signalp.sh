#!/bin/bash
cd "/content/drive/My Drive/MSc Research/installs/signalp-5.0b"
chmod +x "/content/drive/My Drive/MSc Research/installs/signalp-5.0b/bin/signalp"
"/content/drive/My Drive/MSc Research/installs/signalp-5.0b/bin/signalp" -fasta "/content/drive/My Drive/MSc Research/data/alignments/post/human_h3.fasta" -format short -gff3 -prefix output

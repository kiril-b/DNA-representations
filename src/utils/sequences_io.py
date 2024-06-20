from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def read_sequences(file_path: str) -> SeqRecord:
    with open(file_path, "r") as fna_file:
        sequence = next(SeqIO.parse(fna_file, "fasta"))
    return sequence

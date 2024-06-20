from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

HUMAN_PATH = "../data/homo_sapiens_hemoglobin_subunit_beta_gid_3043.fna"
CHIMPANZEE_PATH = "../data/chimpanzee_hemoglobin_subunit_beta_gid_450978.fna"
RABBIT_PATH = "../data/rabbit_hemoglobin_subunit_beta_gid_100009084.fna"


def read_sequences(file_path: str) -> SeqRecord:
    with open(file_path, "r") as fna_file:
        sequence = next(SeqIO.parse(fna_file, "fasta"))
    return sequence
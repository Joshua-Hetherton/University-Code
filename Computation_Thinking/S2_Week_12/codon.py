
from enum import IntEnum
import functools
from typing import List

class Nucleotide(IntEnum):
    """Representation of the four DNA nucleotides."""

    A = 0
    C = 1
    G = 2
    T = 3

import functools


@functools.total_ordering
class Codon:
    """A codon is a sequence of three nucleotides that codes for a specific amino acid."""

    def __init__(self, first: Nucleotide, second: Nucleotide, third: Nucleotide):
        """Initialize a codon with three nucleotides."""
        self.nucleotides = (first, second, third)
    
    def __eq__(self, other: 'Codon') -> bool:
        """Compare two codons for equality."""
        if not isinstance(other, Codon):
            return False
        return self.nucleotides == other.nucleotides
    
    def __lt__(self, other: 'Codon') -> bool:
        """Compare if this codon is less than another (for sorting)."""
        if not isinstance(other, Codon):
            raise TypeError("Can only compare with another Codon")
        return self.nucleotides < other.nucleotides
    
    def __gt__(self, other: 'Codon') -> bool:
        """Compare if this codon is greater than another (for sorting)."""
        if not isinstance(other, Codon):
            raise TypeError("Can only compare with another Codon")
        return self.nucleotides > other.nucleotides
    
    def __repr__(self) -> str:
        """String representation of the codon."""
        return f"Codon({self.nucleotides[0].name}, {self.nucleotides[1].name}, {self.nucleotides[2].name})"
    




class Gene:
    """Representation of a gene as a sequence of codons."""
    
    def __init__(self, codons: List[Codon] = None):
        """Initialize a gene with a list of codons."""
        self.codons = codons if codons is not None else []
    
    @classmethod
    def from_string(cls, dna_string: str) -> 'Gene':
        """Create a Gene instance from a DNA string."""
        if not all(nucleotide in "ACGT" for nucleotide in dna_string):
            raise ValueError("Invalid DNA string. Only A, C, G, T are allowed.")
        
        codons = []
        for i in range(0, len(dna_string), 3):
            if (i + 2) >= len(dna_string):  # don't run off end!
                break
            
            # Initialize codon out of three nucleotides
            first = Nucleotide[dna_string[i]]
            second = Nucleotide[dna_string[i + 1]]
            third = Nucleotide[dna_string[i + 2]]
            codons.append(Codon(first, second, third))
            
        return cls(codons)
    
    def __len__(self) -> int:
        """Return the number of codons in the gene."""
        return len(self.codons)
    
    def __getitem__(self, index: int) -> Codon:
        """Access a codon by index."""
        return self.codons[index]
    
    def linear_search(self, key_codon: Codon) -> bool:
        """
        Search for a specific codon using linear search.
        
        Args:
            key_codon: The codon to search for
            
        Returns:
            True if the codon is found, False otherwise
        """
        for i in range(len(self.codons)):
            current=self.__getitem__(i)

            if(current==key_codon):
                return True
            return False

    
    def sorted_copy(self) -> 'Gene':
        """Return a new Gene with codons sorted."""
        return Gene(sorted(self.codons))
    
    def binary_search(self, key_codon: Codon) -> bool:
        """
        Search for a specific codon using binary search.
        Note: The gene must be sorted for this to work correctly.
        
        Args:
            key_codon: The codon to search for
            
        Returns:
            True if the codon is found, False otherwise
        """
        raise NotImplementedError('TODO')
    



#Test 

gene_str = "ACGTGGCTCTCTAACGTACGTACGTACGGGGTTTATATATACCCTAGGACTCCCTTT"
my_gene = Gene.from_string(gene_str)
acg = Codon(Nucleotide.A, Nucleotide.C, Nucleotide.G)
print(f"Linear search for ACG: {my_gene.linear_search(acg)}")
#print(f"Binary search for ACG: {my_gene.sorted_copy().binary_search(acg)}")
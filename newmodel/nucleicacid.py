"""
This module defines objects that represent basic genetic units.

- class Base
    - subclass DnaBase
    - subclass RnaBase

- class NucleicAcid
    - subclass DnaCode
    - subclass RnaCode

- class Duplex

"""


class Base:
    """
    Represents a generic (DNA/RNA) nucleobase, the basic unit of
    genetic code.

    Attributes
    ----------
    name : str
        Letter associated to base (A, C, G, T, U)
    id : int
        Identifier

    Methods
    -------
    complement_id()
        Returns the id of the complementary (generic) base
    complement_dna()
        Returns the complementing DNA base as a DnaBase object
    complement_rna()
        Returns the complementing RNA base as a RnaBase object
    """

    name_id_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}

    def __init__(self, base_name):
        """
        Parameters
        ----------
        base_name : str
            Letter associated to base (A, C, G, T, U)
        """
        # Checking and assigning base name and id
        base_name = base_name.capitalize()
        if base_name in self.name_id_dict:
            self.name = base_name
            self.id = self.name_id_dict[base_name]
        else:
            raise ValueError(f'{base_name} is not a valid base.')

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def complement_id(self):
        """Returns the id of the complementary (generic) base"""
        return 3 - self.id

    def complement_dna(self):
        """Returns the complementing DNA base as a DnaBase object"""
        c_id = self.complement_id()
        return DnaBase(base_id=c_id)

    def complement_rna(self):
        """Returns the complementing RNA base as a RnaBase object"""
        c_id = self.complement_id()
        return RnaBase(base_id=c_id)


class DnaBase(Base):
    """
    Represents a DNA nucleobase, the basic unit of DNA code.

    Attributes
    ----------
    name : str
        Letter associated to base (A, C, G, T)
    id : int
        Identifier (unique)
    """

    id_name_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    def __init__(self, base_name=None, base_id=None):
        """Accepts either name or id as input (id should be keyword arg)"""
        # From base id to base name
        if base_id is not None:
            if base_id in self.id_name_dict:
                self.id = base_id
                self.name = self.id_name_dict[self.id]
            else:
                raise ValueError(f'{base_id} is not a valid base id')

        # From base name to base id
        else:
            if base_name != 'U':
                Base.__init__(self, base_name)
            else:
                raise ValueError('The base U (uracil) is not a DNA base.')


class RnaBase(Base):
    """
    Represents a RNA nucleobase, the basic unit of RNA code.

    Attributes
    ----------
    name : str
        Letter associated to base (A, C, G, U)
    id : int
        Identifier (unique)
    """

    id_name_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

    def __init__(self, base_name=None, base_id=None):
        """Accepts either name or id as input (id should be keyword arg)"""
        # From base id to base name
        if base_id is not None:
            if base_id in self.id_name_dict:
                self.id = base_id
                self.name = self.id_name_dict[self.id]
            else:
                raise ValueError(f'{base_id} is not a valid base id')

        # From base name to base id
        else:
            if base_name != 'T':
                Base.__init__(self, base_name)
            else:
                raise ValueError('The base T (thymine) is not an RNA base.')


class NucleicAcid:
    """
    Represents a generic (DNA/RNA) genetic sequence.

    Attributes
    ----------
    content : list of Base
        Nucleobase content of sequence
    length : int
        Length of sequence
    direction: int
        Directionality: unspecified (0), 5'-to-3' (1), or 3'-to-5' (2)

    Methods
    -------
    invert()
        Returns the inverse sequence, with switched directionality
    complement_dna()
        Returns the complementing DNA code as a DnaCode object
        (inverts directionality)
    complement_rna()
        Returns the complementing RNA code as a RnaCode object
        (inverts directionality)
    make_double_strand()
        Returns a Duplex of strand and its complement of the same type
    """

    nucleic_acid_type = None

    def __init__(self, genetic_code, direction=0):
        base_name_list = list(genetic_code)
        self.content = [Base(x) for x in base_name_list]
        self.ids = [x.id for x in self.content]
        self.length = len(self.content)
        if direction in [0, 1, 2]:
            self.direction = direction
        else:
            self.direction = 0

    def __str__(self):
        base_list = [x.name for x in self.content]
        code_str = ''.join(base_list)
        if self.direction == 0:
            return code_str
        elif self.direction == 1:
            return '5\'-' + code_str + '-3\''
        elif self.direction == 2:
            return '3\'-' + code_str + '-5\''

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        return self.content[item]

    def __inv_direction(self):
        return (3 - self.direction) % 3

    def invert(self):
        """Returns the inverse sequence, with switched
        directionality."""
        inv_type = type(self)
        content_string = str(self).strip("35'-")
        inv_content = content_string[::-1]
        inv_direction = self.__inv_direction()
        return inv_type(inv_content, inv_direction)

    def enforce_direction(self, new_direction):
        """Returns sequence with enforced directionality"""
        if new_direction in [1, 2]:
            if new_direction + self.direction == 3:
                return self.invert()
            else:
                return self
        else:
            raise ValueError(f"{new_direction} is an invalid direction. " +
                             f"Choose 1 (5'-to-3') or 2 (3'-to-5')")

    def complement_ids(self):
        return [3 - base_id for base_id in self.ids]

    def complement_dna(self):
        """Returns the complementing DNA code as a DnaCode object
        (inverts directionality)"""
        c_ids = self.complement_ids()
        c_base_list = [DnaBase(base_id=base_id).name for base_id in c_ids]
        c_base_str = ''.join(c_base_list)
        c_direction = self.__inv_direction()
        return DnaCode(c_base_str, c_direction)

    def complement_rna(self):
        """Returns the complementing RNA code as a RnaCode object
        (inverts directionality)"""
        c_ids = self.complement_ids()
        c_base_list = [RnaBase(base_id=base_id).name for base_id in c_ids]
        c_base_str = ''.join(c_base_list)
        c_direction = self.__inv_direction()
        return RnaCode(c_base_str, c_direction)

    def make_double_strand(self):
        if self.nucleic_acid_type == 'DNA':
            return Duplex(self, self.complement_dna())
        elif self.nucleic_acid_type == 'RNA':
            return Duplex(self, self.complement_rna())
        else:
            raise ValueError('Nucleic acid type must be specified')


class DnaCode(NucleicAcid):
    """
    Represents a DNA sequence.
    """

    nucleic_acid_type = 'DNA'

    def __init__(self, genetic_code, direction=0):
        NucleicAcid.__init__(self, genetic_code, direction)
        # Overwrite content to contain DnaBase
        self.content = [DnaBase(x.name) for x in self.content]


class RnaCode(NucleicAcid):
    """
    Represents an RNA sequence.
    """

    nucleic_acid_type = 'RNA'

    def __init__(self, genetic_code, direction=0):
        NucleicAcid.__init__(self, genetic_code, direction)
        # Overwrite content to contain RnaBase
        self.content = [RnaBase(x.name) for x in self.content]


class Duplex:
    """
    Represents a nucleic acid duplex: dsDNA, dsRNA or DNA:RNA-duplex

    Attributes
    ----------
    strand1 : DnaCode or RnaCode
        First strand of duplex (DNA/RNA)
    strand2 : DnaCode or RnaCode
        Second strand of duplex (DNA/RNA)
    content : list of tuple of Base
        Base pair content of sequence: (base 1, base 2)
    length : int
        Length of sequence
    complementary : bool
        Indicates whether the duplex is fully matched

    Methods
    -------
    invert()
        Returns the inverse duplex, with switched directionality
    switch()
        Returns duplex with strand1 and strand2 interchanged
    mismatch_positions()
        Returns a list of bool indicating whether each base pair
         in the duplex is mismatched
    mismatch_number()
        Returns the number of mismatching base pairs
    """

    def __init__(self, strand1: NucleicAcid, strand2: NucleicAcid):

        # Check nucleic acid types
        if not (isinstance(strand1, (DnaCode, RnaCode)) and
                isinstance(strand2, (DnaCode, RnaCode)) ):
            raise ValueError('Both strands must have specified nucleic acid'
                             'types (DNA or RNA)')

        # Check strand lengths
        if strand1.length != strand2.length:
            raise ValueError('Both strands must be equally long')

        # Aligning the two strands
        d1 = strand1.direction
        d2 = strand2.direction

        # Raises error if one strand has direction and the other hasn't
        if d1 * d2 == 0 and d1 + d2 > 0:
            raise ValueError('One strand has specified directionality and the '
                             'other has not.')

        # enforcing opposite directionality for strands
        new_d2 = (3 - d1) % 3
        strand2 = strand2.enforce_direction(new_d2)
        self.strand1 = strand1
        self.strand2 = strand2

        self.length = strand1.length
        self.content = [(strand1[bp], strand2[bp]) for bp in range(self.length)]
        self.complementary = (self.mismatch_number() == 0)

    def __repr__(self):
        return f"{str(self.strand1)} ({self.strand1.nucleic_acid_type})\n" +\
               f"{str(self.strand2)} ({self.strand2.nucleic_acid_type})\n"

    def invert(self):
        """Returns the inverse duplex, with switched directionality"""
        return Duplex(self.strand1.invert(), self.strand2.invert())

    def switch(self):
        """Returns duplex with strand1 and strand2 interchanged"""
        return Duplex(self.strand2, self.strand1)

    def mismatch_positions(self):
        """Returns a list of bool indicating whether each base pair
        in the duplex is mismatched"""
        return [b1.complement_id() != b2.id for (b1, b2) in self.content]

    def mismatch_number(self):
        """Returns the number of mismatching base pairs"""
        return sum(self.mismatch_positions())

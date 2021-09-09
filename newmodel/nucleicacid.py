"""
This module defines objects that represent basic genetic units.

- class Base
    - subclass DnaBase
    - subclass RnaBase
- class NucleicAcid
    - subclass DnaCode
    - subclass RnaCode
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
    content : list[Base]
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
    """

    def __init__(self, genetic_code, direction=0):
        base_name_list = list(genetic_code)
        self.content = [Base(x) for x in base_name_list]
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

    def __inv_direction(self):
        return (3 - self.direction) % 3

    def __complement_ids(self):
        return [x.complement_id() for x in self.content]
        # perhaps this can be sped up by performing a list operation

    def invert(self):
        """Returns the inverse sequence, with switched directionality"""
        inv_type = type(self)
        content_string = str(self).strip("35'-")
        inv_content = content_string[::-1]
        inv_direction = self.__inv_direction()
        return inv_type(inv_content, inv_direction)

    def complement_dna(self):
        """Returns the complementing DNA code as a DnaCode object
        (inverts directionality)"""
        c_ids = self.__complement_ids()
        c_base_list = [DnaBase(base_id=base_id).name for base_id in c_ids]
        c_base_str = ''.join(c_base_list)
        c_direction = self.__inv_direction()
        return DnaCode(c_base_str, c_direction)

    def complement_rna(self):
        """Returns the complementing RNA code as a RnaCode object
        (inverts directionality)"""
        c_ids = self.__complement_ids()
        c_base_list = [RnaBase(base_id=base_id).name for base_id in c_ids]
        c_base_str = ''.join(c_base_list)
        c_direction = self.__inv_direction()
        return RnaCode(c_base_str, c_direction)


class DnaCode(NucleicAcid):
    """
    Represents a DNA sequence.
    """

    def __init__(self, genetic_code, direction=0):
        NucleicAcid.__init__(self, genetic_code, direction)
        # Overwrite content to contain DnaBase
        self.content = [DnaBase(x.name) for x in self.content]


class RnaCode(NucleicAcid):
    """
    Represents a RNA sequence.
    """

    def __init__(self, genetic_code, direction=0):
        NucleicAcid.__init__(self, genetic_code, direction)
        # Overwrite content to contain RnaBase
        self.content = [RnaBase(x.name) for x in self.content]

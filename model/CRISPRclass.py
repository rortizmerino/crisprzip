class CRISPR():
    def __init__(self,guide_length,PAM_length, PAM_sequence,
                 PAM_direction='5 to 3' ):
        self.guide_length = guide_length
        self.PAM_length = PAM_length
        self.target_length = self.guide_length + self.PAM_length
        self.PAM = PAM_sequence
        self.PAM_direction = PAM_direction

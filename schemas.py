class InputExample():
    
    def __init__(self, guid, text, label=None, segment_ids=None):
        self.guid = guid
        self.text = text
        self.label = label
        self.segment_ids = segment_ids
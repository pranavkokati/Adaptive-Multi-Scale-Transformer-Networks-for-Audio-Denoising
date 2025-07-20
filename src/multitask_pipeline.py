class MultiTaskEnhancer:
    def __init__(self, tasks=None):
        if tasks is None:
            tasks = ['denoise', 'dereverb', 'declip']
        self.tasks = tasks
    def process(self, audio, task='denoise'):
        # Placeholder: just return input
        assert task in self.tasks
        return audio 
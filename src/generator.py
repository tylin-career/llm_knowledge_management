class Generator:
    def __init__(self, config):
        self.config = config

    def generate(self):
        print('Generating...')
        print(f'Config: {self.config}')
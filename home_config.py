class Configuration:
    def __init__(self, model, openai_api_key, openai_api_base, temperature):
        self.model = model
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base
        self.temperature = temperature
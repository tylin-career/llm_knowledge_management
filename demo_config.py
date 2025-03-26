class Configuration:
    def __init__(self):
        self.debug = True


    def _update_config(self):
        self.debug = False
        print('Config updated')
class Auth:
    def __init__(self, username, password):
        self.username = username
        self.password = password

    def login(self):
        if self.username == "biguser" and self.password == "npspo":
            return True
        else:
            return False

    def logout(self):
        return False
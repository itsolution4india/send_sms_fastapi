import string
import secrets


# Token generation function
def generate_token(length=100):
    
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    prefix = 'ITS'
    token = prefix + ''.join(secrets.choice(alphabet) for _ in range(length))
    return token
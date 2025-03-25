import string
import secrets


# Token generation function
def generate_token(length=120):
    
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    prefix = 'ITS'
    token = prefix + ''.join(secrets.choice(alphabet) for _ in range(length))
    return token

# Generate random message ID
def generate_message_id():
    """
    Generate a random message ID similar to the example
    """
    return int(''.join(secrets.choice(string.digits) for _ in range(15)))
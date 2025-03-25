import string
import secrets


# Token generation function
def generate_token(length: int = 120) -> str:
    """
    Generate a long, secure token using a mix of uppercase, lowercase, digits, and punctuation.
    Mimics the format of Facebook/Meta-like access tokens.
    """
    # Characters to use for token generation
    characters = string.ascii_uppercase + string.ascii_lowercase + string.digits + '!@#$%^&*()_+-=[]{}|;:,.<>?'
    
    # Generate token with a mix of complexity
    token = ''.join(secrets.choice(characters) for _ in range(length))
    
    # Ensure the token starts with a format similar to the example
    prefix = 'ITS'
    token = prefix + token[len(prefix):]
    
    return token


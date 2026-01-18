from fastapi.testclient import TestClient
from pathlib import Path
import sys #TODO - packge?
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)
import dotenv
import os
dotenv.load_dotenv(os.path.join(project_root, '.env'))

# TODO - tow attempts 
#TODO - system test
from src.api.app import app

client = TestClient(app)
### chack get 
def test_info():
    response = client.get("/",headers={
                    "username": os.getenv('USERNAME'), 
                    "password": os.getenv('PASSWORD')
                })
    assert response.status_code == 200
    assert response.json() == {"instructions": 
            """Please provide a JSON file in the following format: 

            {

                "string": str - string with only alphabetic characters

            }
            
            and enter username and password as an header
            """}
### chack wrong address
def test_wrong_address():
    """Test response from wrong address"""
    response = client.post(
            "/newaddress", 
            json={"string": 'aba'},
            headers={
            "username": os.getenv('USERNAME'), 
            "password": os.getenv('PASSWORD')}
            )
    assert response.status_code == 404
    assert 'Not Found' in response.json()["detail"]

### chack wrong credential - both
def test_wrong_credential():
    '''Test API access with incorrect credential'''
    response = client.post(
                "/string", 
                json={"string": 'aba'},
                headers={
                    "username": 'wrong', 
                    "password": 'wrong'
                }
            )
    assert response.status_code == 401
    assert response.json() == {"detail": "wrong credentials"}

### chack wrong credential - username 
def test_wrong_credential_username():
    '''Test API access with incorrect credential'''
    response = client.post(
                "/string", 
                json={"string": 'aba'},
                headers={
                    "username": 'wrong', 
                    "password": os.getenv('PASSWORD')
                }
            )
    assert response.status_code == 401
    assert response.json() == {"detail": "wrong credentials"}


### chack wrong credential - password 
def test_wrong_credential_password():
    '''Test API access with incorrect credential'''
    response = client.post(
                "/string", 
                json={"string": 'aba'},
                headers={
                    "username": os.getenv('USERNAME'), 
                    "password": 'wrong'
                }
            )
    assert response.status_code == 401
    assert response.json() == {"detail": "wrong credentials"} 

### chack wrong input type 
def test_wrong_input_type():
    response = client.post(
        "/string", 
        content="not a json",
        headers={
            "username": os.getenv('USERNAME'), 
            "password": os.getenv('PASSWORD')})
    assert response.status_code == 422
    assert response.json()["detail"][0]['msg'] == "JSON decode error"

### chack wrong input - json without the right key 
def test_json_without_string_key():
        """Test JSON input without 'string' key"""
        response = client.post(
            "/string", 
            json={"text": "hello"},
            headers={
            "username": os.getenv('USERNAME'), 
            "password": os.getenv('PASSWORD')}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]['msg'] in ['field required','Field required']

### chack wrong input type - non string input
def test_non_string_string_value():
    """Test JSON input with non-string 'string' value"""
    response = client.post(
            "/string", 
            json={"string": 12345},
            headers={
            "username": os.getenv('USERNAME'), 
            "password": os.getenv('PASSWORD')}
            )
    assert response.status_code == 422
    assert response.json()["detail"][0]['msg'] in ['"string" must be a string','Value error, "string" must be a string']

### chack legitimate cases
def test_normal_palindrome_cases():
    """Test various palindrome scenarios"""
    test_cases = [
            ("qwerewq", 7),
            ("hello", 2),
            ("abba", 4),
            ("abcba", 5),
            ("abarra",4),
            ("abcdefedcdefedc",13)]
        
    for test_string, expected_length in test_cases:
        response = client.post(
            "/string", 
            json={"string": test_string},
            headers={
                "username": os.getenv('USERNAME'), 
                "password": os.getenv('PASSWORD')
            }
        )
        assert response.status_code == 200
        assert response.json() == {"answer": expected_length}

### chack legitimate case - empty palindrome
def test_empty_palindrome():
    response = client.post(
                "/string", 
                json={"string": ''},
                headers={
                    "username": os.getenv('USERNAME'), 
                    "password": os.getenv('PASSWORD')
                }
            )
    assert response.status_code == 200
    assert response.json() == {"answer": 0}

### chack legitimate case - 0 length palindrome with symbols 
def test_symbols_palindrome():
    response = client.post(
                "/string", 
                json={"string": '!#$ '},
                headers={
                    "username": os.getenv('USERNAME'), 
                    "password": os.getenv('PASSWORD')
                }
            )
    assert response.status_code == 200
    assert response.json() == {"answer": 0}

### chack legitimate case - 0 length palindrome with numbers 
def test_numbers_palindrome():
    response = client.post(
                "/string", 
                json={"string": '12345'},
                headers={
                    "username": os.getenv('USERNAME'), 
                    "password": os.getenv('PASSWORD')
                }
            )
    assert response.status_code == 200
    assert response.json() == {"answer": 0}

### chack legitimate case -  palindrome with spaces 
def test_string_with_spaces():
    response = client.post(
                "/string", 
                json={"string": 'a b a'},
                headers={
                    "username": os.getenv('USERNAME'), 
                    "password": os.getenv('PASSWORD')
                }
            )
    assert response.status_code == 200
    assert response.json() == {"answer": 3}


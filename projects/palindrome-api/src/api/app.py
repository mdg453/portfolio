from fastapi import FastAPI, Request,HTTPException,Header,Depends
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import sys
import time
from pydantic import BaseModel,Extra,validator,ValidationError #to ensure the integrity of the input

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)
from config.config import maximise_time,credentials_type,log_mode,logger_name
from src.palindrome import pal_length_v2,pal_length_manachers,clean_string
from logger import Logger
import dotenv
if credentials_type =='env':
    dotenv.load_dotenv(os.path.join(project_root, '.env'))
if credentials_type =='docker_env':
    dotenv.load_dotenv()
from src.api.credentials import check_pass_name

# Create a Logger instance
logger = Logger(logger_name=logger_name,log_mode=log_mode,logs_dir=project_root)
#for now i save the logs at project_root\logs. it is not best practice. 
# In real project i will save them at the rellevent folder/db.

app = FastAPI()
class api_input(BaseModel):
    string: str

    class Config:
        extra = Extra.ignore 

    @validator('string', pre=True)
    def validate_string_type(cls, v):
        # Explicitly convert input to string and validate
        if not isinstance(v, str):
            logger.error(f"Invalid input type: {type(v)}, Input value: {v}")
            raise ValueError('"string" must be a string')
        return v
    
# TODO - it may be a good idea to use https. unforthnaly it is not very straightforward in fastapi.
# see https://fastapi.tiangolo.com/deployment/https/#dns 
@app.middleware("http")    
async def authenticator(request: Request, call_next):
    """
    Middleware to authenticate incoming requests.
    Checks for valid username and password in request headers.
    return 401 if credentials are missing or incorrect. else - activate palindrom_length.
    
    """
    username = request.headers.get('username')
    password = request.headers.get('password')
    logger.info(f"Incoming request to {request.url} - checking authentication")
    if (not username) or (not password) or not check_pass_name(username, password):
        logger.warning(f"Authentication failed for request to {request.url}")
        return JSONResponse(
            status_code=401, 
            content={"detail": "wrong credentials"}
        )
    
    response = await call_next(request)
    return response

@app.post("/string")
async def palindrom_length(input:api_input,request: Request,maximise_time = maximise_time):
    """
    Calculate max palindrome lengths in a substring.
    Expects JSON input with a 'string' key.
     Returns lengths of longest palindromic substrings.

     inputs:
     - request (Request): Incoming HTTP request with JSON body
     - maximise_time: Does the user want to minimize runtime (True) or memory usage (False)
    """
    
    try:
        input_string = input.string
        logger.debug(f'the input is {input_string}')
    except ValueError:
        logger.error(f"Error processing raw input: {e}")
        raise HTTPException(status_code=400, detail='Invalid JSON')  
    
    start_time = time.time()  
    try:
        string_clean = clean_string(input_string)
        logger.info(f"Processing cleaned string: {string_clean}")
        answer = pal_length_manachers(string_clean) if maximise_time else pal_length_v2(string_clean)
        func_processing_time = time.time() - start_time
        logger.info(f'the function process {input_string} in {func_processing_time:.7f}')
        return {'answer':answer} 
    except Exception as e:
        logger.error(f"Error processing palindrome: {e}")
        raise HTTPException(status_code=500, detail='Internal server error')

@app.get("/")
def read_root():
    """
    a get request 
    return: instructions on how to use this API 
    """
    logger.info("Root endpoint accessed - returning usage instructions")
    return {"instructions": 
            """Please provide a JSON file in the following format: 

            {

                "string": str - string with only alphabetic characters

            }
            
            and enter username and password as an header
            """}



# Palindrome Length Finder

## Project Overview
A FastAPI-based web service that calculates the length of the longest palindromic substring using two different algorithms.

## Algorithms
1. **Expand Around Center**: 
   - Time Complexity: O(n²)
   - space complexity: o(1)
   - Explores all possible center points
   - Expands outwards to find palindromes

2. **Manacher's Algorithm**: 
   - Time Complexity: O(n)
   - space complexity: o(n)
   - Reduces redundant comparisons
   - for deeper explanation see https://www.geeksforgeeks.org/manachers-algorithm-linear-time-longest-palindromic-substring-part-1/ 

## TL;DR - Running the Application

### Local Development
```bash
uvicorn src.api.app:app --reload
```

### Docker Deployment
1. Create logs volume:
```bash
docker volume create my-logs-volume
```

2. Run Docker container:
```bash
docker run -p 8000:8000  -e USERNAME=your_username  -e PASSWORD=your_password -v my-logs-volume:/app/logs palindrom-api
```

## API Endpoints
- `/string` (POST)
  - Accepts JSON with a 'string' parameter
  - Returns longest palindrome lengths using the method specify in the config file

- `/` (GET)
  - Explains how to access the API

## Authentication
- Requires HTTP headers:
  - `username`
  - `password`

### Authentication Methods

#### 1. Environment File (.env)
- Create `.env` file:
```
USERNAME=your_username
PASSWORD=your_password
```
- Set in `config.py`: `credential_type = 'env'`

#### 2. Docker Environment Variables
```bash
docker run -p 8000:8000 \
  -v my-logs-volume:/app/logs \
  -e USERNAME=your_username \
  -e PASSWORD=your_password \
  palindrom-api
```
- Set in `config.py`: `credential_type = 'docker_env'`

#### 3. Docker Swarm Secrets
1. Initialize Swarm:
```bash
docker swarm init
```

2. Create Secrets:
```bash
echo "your_username" | docker secret create username -
echo "your_password" | docker secret create password -
```

3. Run Container:
```bash
docker run -p 8000:8000 \
  -v my-logs-volume:/app/logs \
  --secret username \
  --secret password \
  palindrom-api
```
- Set in `config.py`: `credential_type = 'docker_swarm'`

## Logging
- Logs are saved daily in the `logs` directory
- Each log file is named with the format: `{logger_name}-{YYYYMMDD}.log`
- Log files include timestamp, logger name, log level, and message

## Testing
1. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Run tests using pytest:
```bash
pytest test_api.py
pytest test_palindrome.py
```



## Configuration
- Modify `config.py` to choose credential method, log mode, time or space minimization and logger name:

```python
# Credential type
credentials_type = 'env'  # Options: 'env', 'docker_env', 'docker_swarm'

# Logging mode
log_mode = 'DEBUG'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'

# Logger name
logger_name = 'palindrome_api'

# Algorithm optimization
maximise_time = True  # True: minimize runtime, False: minimize memory usage
```

## Setup
1. Install dependencies
2. Configure `config.py` with.
3. create credential an expleined above.
3. Run as explain above. 


## CI/CD Pipeline

### GitHub Actions Workflow
The project uses GitHub Actions for continuous integration and deployment:

#### Workflow Triggers
- Triggered on:
  - Push to `main` branch
  - Pull requests to `main` branch

#### Workflow Steps
1. **Code Checkout**
   - Uses GitHub Actions to check out the repository code

2. **Python Setup**
   - Sets up Python 3.10.12 environment

3. **Dependency Installation**
   - Upgrades pip
   - Installs project requirements
   - Installs additional testing dependencies

4. **Testing**
   - Runs tests using pytest
   - Uses environment secrets for authentication
   - Sets PYTHONPATH to ensure test discovery

5. **Docker Image Build**
   - Builds Docker image with tag `yotam433/palindrome-api:latest`

6. **Docker Hub Deployment**
   - Logs in to Docker Hub
   - Pushes image to Docker Hub
   - Only occurs on direct pushes to `main` branch

## Requirements
- Python 3.8+
- fastapi==0.110.3
- uvicorn==0.23.2
- python-dotenv==1.0.0
- pytest==7.4.0
- httpx==0.26.0
- regex==2024.9.11
- pydantic ==2.7.4

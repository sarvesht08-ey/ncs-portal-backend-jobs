import os
import logging
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load .env file (put sensitive info there)
load_dotenv()

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("jobsearch_api")

# ----------------------------
# Database Configuration
# ----------------------------
DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    logger.warning("DATABASE_URL is not set in environment variables!")

# ----------------------------
# Azure OpenAI Configuration
# ----------------------------
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/") + "/"
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2023-05-15")
AZURE_GPT_DEPLOYMENT = os.getenv("AZURE_GPT_DEPLOYMENT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

if not AZURE_API_KEY or not AZURE_ENDPOINT:
    logger.warning("Azure OpenAI credentials are missing!")

# Azure client (shared instance)
azure_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

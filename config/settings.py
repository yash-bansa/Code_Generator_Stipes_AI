import os
from dotenv import load_dotenv
from pathlib import Path
import redis

# Load environment variables from .env file
load_dotenv()

class Settings:
    # ==============================================
    # PROVIDER SELECTION
    # ==============================================
    LM_CLIENT_PROVIDER = os.getenv("LM_CLIENT_PROVIDER", "tiger")




    # ==============================================
    # TIGER ANALYTICS CONFIGURATION
    # ==============================================
    TIGER_BASE_URL = os.getenv("TIGER_BASE_URL", "https://api.ai-gateway.tigeranalytics.com")
    TIGER_API_KEY = os.getenv("TIGER_API_KEY", "")
    TIGER_MODEL_NAME = os.getenv("TIGER_MODEL_NAME", "gpt-4o")

    # ==============================================
    # BACKWARD COMPATIBILITY
    # ==============================================
    # Keep MODEL_NAME for backward compatibility
    MODEL_NAME = os.getenv("MODEL_NAME", "claude-3.7-sonnet")

    # ==============================================
    # PROJECT SETTINGS
    # ==============================================
    # PROJECT_ROOT_PATH = Path(os.getenv("PROJECT_ROOT_PATH", "./examples/sample_project"))
    # OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "./output/generated_code"))
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 1048576))  # 1MB
    SUPPORTED_EXTENSIONS = os.getenv("SUPPORTED_EXTENSIONS", ".py,.json,.yaml,.yml,.txt,.md").split(",")

    # ==============================================
    # AGENT SETTINGS
    # ==============================================
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", 180))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4000))

    # ==============================================
    # ADDITIONAL TIGER ANALYTICS SETTINGS
    # ==============================================
    TIGER_ORGANIZATION_ID = os.getenv("TIGER_ORGANIZATION_ID", "")
    TIGER_USER_ID = os.getenv("TIGER_USER_ID", "")

  

    # ==============================================
    # REDIS CONFIGURATION
    # ==============================================
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB = int(os.getenv("REDIS_DB", 0))

    @classmethod
    def get_redis_connection(cls):
        """Get a Redis connection object"""

        return redis.StrictRedis(
            host=cls.REDIS_HOST,
            port=cls.REDIS_PORT,
            password=cls.REDIS_PASSWORD or None,
            db=cls.REDIS_DB,
            decode_responses=True,
            ssl = True,
        )

    @classmethod
    def get_current_provider_config(cls):
        """Get configuration for currently selected provider"""
        provider = cls.LM_CLIENT_PROVIDER.lower()

        if provider == "tiger":
            return {
                "provider": "tiger",
                "base_url": cls.TIGER_BASE_URL,
                "api_key": cls.TIGER_API_KEY,
                "model_name": cls.TIGER_MODEL_NAME,
                "requires_auth": True
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @classmethod
    def validate_configuration(cls):
        """Validate current configuration"""
        errors = []
        warnings = []

        provider = cls.LM_CLIENT_PROVIDER.lower()

        if provider == "tiger":
            if not cls.TIGER_API_KEY:
                errors.append("TIGER_API_KEY is required for Tiger Analytics provider")
            elif not cls.TIGER_API_KEY.startswith("sk-"):
                warnings.append("Tiger Analytics API key should start with 'sk-'")

            if not cls.TIGER_BASE_URL:
                errors.append("TIGER_BASE_URL is required for Tiger Analytics provider")

        else:
            errors.append(f"Unsupported provider: {provider}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    @classmethod
    def get_model_name_for_provider(cls):
        """Get the correct model name for current provider"""
        provider = cls.LM_CLIENT_PROVIDER.lower()

        if provider == "tiger":
            return cls.TIGER_MODEL_NAME
        else:
            return cls.MODEL_NAME  # Fallback

    
# Create global settings instance
settings = Settings()

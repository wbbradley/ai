import os
import subprocess
import sys
import tomllib
from pathlib import Path
from tomllib import TOMLDecodeError
from typing import Any, Optional, cast

from pydantic import BaseModel


class AIConfigError(Exception):
    """All config related errors."""


class InvalidConfigurationError(AIConfigError):
    """General failure reading config."""


class InvalidProviderModelError(AIConfigError):
    """General failure reading config."""


class MissingConfigurationError(AIConfigError):
    """Missing configuration."""


class UnexpectedConfigurationError(AIConfigError):
    """Unexpected error in configuration."""


class HasAPIKey(BaseModel):
    api_key: Optional[str] = None
    api_key_cmd: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.api_key_cmd is not None:
            try:
                self.api_key = subprocess.check_output(self.api_key_cmd, shell=True).strip().decode("utf-8")
            except subprocess.CalledProcessError as e:
                raise InvalidConfigurationError("openai.api_key_cmd returned an error.") from e
            self.api_key_cmd = None


class OpenAIConfig(HasAPIKey):
    model: str


class AnthropicConfig(HasAPIKey):
    model: str


class Config(BaseModel):
    transcript_dir: str
    report_dir: str
    provider: str
    openai: Optional[OpenAIConfig] = None
    anthropic: Optional[AnthropicConfig] = None

    def get_provider_model(self, provider: str) -> str:
        provider_config = getattr(self, provider, None)
        if not provider_config:
            raise InvalidProviderModelError(provider)
        return cast(str, getattr(provider_config, "model"))

    def model_post_init(self, __context: Any) -> None:
        # Allow for usage '~' in config
        self.transcript_dir = os.path.expanduser(self.transcript_dir)
        self.report_dir = os.path.expanduser(self.report_dir)
        # Ensure the config directories exist.
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.transcript_dir, exist_ok=True)


def load_config_toml(filename: Path) -> Config:
    """Load the config into the Config data structure."""
    try:
        with filename.open("rb") as f:
            data = tomllib.load(f)
            return Config(**data)

    except TypeError as e:
        raise UnexpectedConfigurationError(str(e))
    except TOMLDecodeError as e:
        print(f"[ai] failed to parse {filename}: {e}", file=sys.stderr)
        raise InvalidConfigurationError(f"invalid configuration in {filename}: {e}")


def fetch_config() -> Config:
    current_dir = Path(os.getcwd())
    checked_paths = set()
    while current_dir:
        filename = Path(current_dir) / ".ai.toml"
        if filename.exists():
            if config := load_config_toml(filename):
                return config
        else:
            checked_paths.add(filename)
        if current_dir != current_dir.parent:
            current_dir = current_dir.parent
        else:
            break
    filename = Path.home() / ".ai.toml"
    if filename not in checked_paths and filename.exists():
        if config := load_config_toml(filename):
            return config
    raise MissingConfigurationError()

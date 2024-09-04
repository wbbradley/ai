import logging
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from tomllib import TOMLDecodeError
from typing import Any, Optional, cast

from pydantic import BaseModel

from ai.colors import ColorScheme, erase_line
from ai.output import note


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
                note(f"\rrunning api_key_cmd '{self.api_key_cmd}'...", end="")
                self.api_key = subprocess.check_output(self.api_key_cmd, shell=True).strip().decode("utf-8")
                # Erase line.
                erase_line()
            except subprocess.CalledProcessError as e:
                raise InvalidConfigurationError(f"api_key_cmd '{self.api_key_cmd}' returned an error.") from e
            self.api_key_cmd = None


class OpenAIConfig(HasAPIKey):
    model: str


class AnthropicConfig(HasAPIKey):
    model: str


def translate_log_level(log_level: str) -> int:
    return cast(int, getattr(logging, log_level.upper()))


class Config(BaseModel):
    colorscheme: ColorScheme = ColorScheme()
    log_filename: Optional[str] = "~/ai.log"
    log_level: str = "warn"
    log_level_int: int = logging.WARN
    transcript_dir: str
    report_dir: str
    system_prompt: str = ""
    provider: str
    temperature: float = 1.0
    max_tokens: int = 1024
    openai: Optional[OpenAIConfig] = None
    anthropic: Optional[AnthropicConfig] = None
    format_as_markdown: bool = True

    def get_provider_model(self, provider_override: Optional[str] = None) -> str:
        provider: str = provider_override or self.provider
        provider_config = getattr(self, provider, None)
        if not provider_config:
            raise InvalidProviderModelError(provider)
        return cast(str, getattr(provider_config, "model"))

    def model_post_init(self, __context: Any) -> None:
        self.log_level_int = translate_log_level(self.log_level)
        # Allow for usage '~' in config
        self.transcript_dir = os.path.expanduser(self.transcript_dir)
        self.report_dir = os.path.expanduser(self.report_dir)

        if self.log_filename is not None:
            self.log_filename = os.path.expanduser(self.log_filename)
            os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)

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

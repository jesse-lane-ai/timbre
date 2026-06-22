"""Shared error types and exit codes.

Exit codes: 0 success · 1 bad input · 3 engine error
"""

EXIT_OK = 0
EXIT_BAD_INPUT = 1
EXIT_ENGINE_ERROR = 3


class TimbreError(Exception):
    """Base error carrying a JSON-envelope message and a process exit code."""

    code = EXIT_BAD_INPUT

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class BadInputError(TimbreError):
    code = EXIT_BAD_INPUT


class EngineError(TimbreError):
    code = EXIT_ENGINE_ERROR

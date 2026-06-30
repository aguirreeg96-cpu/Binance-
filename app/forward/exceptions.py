class ForwardError(Exception):
    """Base class for forward paper-trading errors."""


class ForwardConfigMismatchError(ForwardError):
    """Raised when the frozen manifest hash does not match the launch on record.

    The frozen B_4h configuration must never change after launch. If the
    computed hash differs from what was recorded at first launch, the
    engine refuses to start rather than silently trading under a different
    configuration.
    """


class ForwardStateInconsistentError(ForwardError):
    """Raised when recovered DB state cannot be reconciled.

    On this error the engine must not open or close any position; it logs
    a SystemEvent and requires manual intervention before evaluation can
    resume.
    """


class ForwardDataGapError(ForwardError):
    """Raised when required candle data is missing beyond the grace period."""

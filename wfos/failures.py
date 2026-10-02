"""The failure vocabularies, in a leaf module.

Two of them, and they stay two.

`WIRE_CODES` classifies a **model-interface** failure: the answer was cut off, the
provider refused, the credential was rejected. `FAILURE_CLASSES` classifies a
**state outcome**, and it is those wire codes plus the workflow's own verdicts —
the build broke, the tests failed, the regression came back.

They are not merged because they are at different granularities. One failed state
can contain ten tool calls with three different codes, and a state that never
failed can contain a refusal; a single enum would force a projection rule ("which
code wins?") and that rule would *become* the semantics.

They live here, outside `harness/` and `llm/`, because both of those need them and
so does `eval/` — and an evaluator importing the orchestrator to reach a
vocabulary is an evaluator that will grow a dependency on the runner's internals.
The originals re-export from here, so no caller had to move.
"""
import errno

# Why a *model call* did not produce an answer. Split out of the old catch-all
# `agent_no_output` because they need opposite responses: a truncated answer wants
# more room, a malformed one wants to be re-asked, an auth failure wants a human.
WIRE_OUTPUT_TRUNCATED = "output_truncated"
WIRE_OUTPUT_MALFORMED = "output_malformed"
WIRE_PROVIDER_REFUSED = "provider_refused"
WIRE_RATE_LIMITED = "rate_limited"
WIRE_AUTH_FAILED = "auth_failed"
WIRE_PROVIDER_ERROR = "provider_error"

WIRE_CODES = (WIRE_OUTPUT_TRUNCATED, WIRE_OUTPUT_MALFORMED, WIRE_PROVIDER_REFUSED,
              WIRE_RATE_LIMITED, WIRE_AUTH_FAILED, WIRE_PROVIDER_ERROR)

# The local failures. Named here rather than inline so the classifier and the
# vocabulary cannot drift apart, and so a reader wondering "what else can a run
# fail with" finds them in the one tuple that answers that.
FAILURE_PERMISSION = "permission_error"
FAILURE_DISK = "disk_error"
FAILURE_GENERIC_OS = "generic_os_error"

# A disk that is full, read-only, or giving I/O errors. Deliberately short: an
# errno nobody has thought about is better reported as a generic OS error than
# guessed into a bucket that suggests a remedy.
_DISK_ERRNOS = frozenset(filter(None, (
    errno.ENOSPC,                                  # no space left
    errno.EROFS,                                   # read-only filesystem
    errno.EIO,                                     # I/O error
    errno.EDQUOT if hasattr(errno, "EDQUOT") else None,   # quota exceeded
)))

# The errnos that really are the network. `ConnectionError` and `TimeoutError`
# cover the rest by type, because Windows and POSIX disagree about which errno a
# dropped connection reports and the exception class is the portable answer.
_NETWORK_ERRNOS = frozenset(filter(None, (
    errno.ECONNREFUSED, errno.ECONNRESET, errno.ECONNABORTED, errno.ETIMEDOUT,
    errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EHOSTDOWN, errno.EPIPE,
    errno.EAGAIN if hasattr(errno, "EAGAIN") else None,
    errno.EAI_AGAIN if hasattr(errno, "EAI_AGAIN") else None,
)))

# Why a *state* did not succeed. Retries are budgeted per (state, class) so that
# the same failure recurring is visible instead of being averaged away, and a
# repeated class can escalate to a human rather than burning the retry budget.
FAILURE_CLASSES = ("build_error", "test_failure", "regression",
                   "verification_failed", "no_change",
                   # the agent produced nothing usable — recorded by the Harness
                   # rather than derived from an output, because there is none
                   "agent_no_output",
                   # raised out of a state: a provider that could not be reached,
                   # and anything else genuinely unanticipated
                   "network_error", "unexpected_error",
                   # The local side of the same question. These used to be folded
                   # into `network_error` because the classifier asked one binary
                   # question — "is this an OSError?" — and a full disk, a
                   # read-only tree and an unreachable host all answered yes. They
                   # want different responses: a disk wants space, a permission
                   # wants a human, a network wants a retry.
                   FAILURE_PERMISSION, FAILURE_DISK, FAILURE_GENERIC_OS,
                   *WIRE_CODES)


def classify_os_error(exc: BaseException) -> str:
    """Which *local* failure an `OSError` is — never `unexpected_error`.

    Reads `errno`, never the message text: the same rule the wire vocabulary
    follows, and for the same reason — a message is prose that changes, an errno
    is a number the operating system promises.

    Anything that is not an `OSError` at all is `unexpected_error`, so a caller
    can use this as the last branch of a chain without a second type check.
    """
    if not isinstance(exc, OSError):
        return "unexpected_error"
    if isinstance(exc, PermissionError):        # EACCES / EPERM, and Windows' own
        return FAILURE_PERMISSION
    code = exc.errno
    if code in _DISK_ERRNOS:
        return FAILURE_DISK
    if code in _NETWORK_ERRNOS or isinstance(exc, (ConnectionError, TimeoutError)):
        return "network_error"
    return FAILURE_GENERIC_OS

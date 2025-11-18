"""Common benchmarking errors."""

class BenchmarkError(Exception):
    pass

class BenchmarkNotConfiguredError(BenchmarkError):
    pass

class BenchmarkRunningError(BenchmarkError):
    pass

class BenchmarkNotRunningError(BenchmarkError):
    pass

class BenchmarkNotInCollectionData(BenchmarkError):
    pass

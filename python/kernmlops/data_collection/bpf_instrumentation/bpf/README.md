# Kermit Event Transport

Every event producer uses the same reserve/fill/submit API:

```c
#include "kermit.h"

struct event_data {
  u64 timestamp;
};

KERMIT_OUTPUT(events, struct event_data);

// Inside the probe:
struct event_data* data = KERMIT_RESERVE(events, struct event_data, 1);
if (data) {
  data->timestamp = bpf_ktime_get_ns();
  KERMIT_SUBMIT(events, ctx, data);
}
// Keep state cleanup outside the reservation-success branch.
```

With `-DKERMIT_CAPTURE`, reservation provides zero-initialized stack storage and
submission uses the existing perf buffer. Event layouts, CPU metadata, Python
callbacks, and polling are unchanged. The capture runner's hooks use
`capture_bpf()` to supply this flag automatically.

Without that flag, reservation returns a pointer into a mmapable MPMO array slot.
Fill the payload in place, then submit to publish it. There is no temporary event
copy. Initialize every payload field before submitting, and submit every
successfully reserved slot. Reservation failure must not skip probe cleanup.

The queue maps are `mpmo_head_<name>` and `mpmo_buff_<name>`. Each buffer element
contains an aligned `u64 valid` followed by the event payload `data`. Capacity
defaults to 4096 slots per output, configurable with `-DKERMIT_QUEUE_SIZE=N`.
This is a lossy overwrite queue: unread slots can be reused. Slots owned by a
reader or writer are not overwritten; failed claims return `NULL`.

Readers must atomically claim `READABLE (1)` as `READING (2)` with acquire
ordering before accessing the payload, then release it as `INVALID (0)`.
Producers claim `INVALID` or `READABLE` as `WRITING (3)` and release-publish as
`READABLE`. The head counts reservation attempts, not committed records.

BCC rejects map-method calls inside macros. Use `compile_bpf()` for non-capture
builds or `capture_bpf()` for capture builds: these expand the Kermit macros
before BCC's rewrite pass, leaving kernel includes and conditionals to BCC.
This requires `cpp`, available in the project container. MPMO also requires
kernel/compiler support for mmapable BPF arrays and 64-bit BPF atomic operations.

Run transport tests in the project container:

```sh
make docker INTERACTIVE=i CONTAINER_CMD="env PYTHONPATH=python/kernmlops python -m unittest discover -s python/tests -v"
```

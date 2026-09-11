"""Run in the privileged project container with PYTHONPATH=python/kernmlops."""

import ctypes as ct
import mmap
import os
import unittest

from data_collection.bpf_instrumentation.bpf_hook import capture_bpf, compile_bpf

SOURCE = r"""
#include <linux/version.h>
#include "kermit.h"

#define PAYLOAD_VALUE (40 + \
                       2)
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 5, 0)
#error "The MPMO backend requires mmapable BPF arrays"
#endif

struct sample {
  u32 pid;
  u64 value;
  u32 unused;
};

KERMIT_OUTPUT(events, struct sample);
BPF_ARRAY(completed, u64, 1);

int emit(void* ctx) {
  u64 pid = bpf_get_current_pid_tgid();
  if ((pid >> 32) != TARGET_PID)
    return 0;
  struct sample* data = KERMIT_RESERVE(events, struct sample, RESERVE_ATTEMPTS);
  if (data) {
    data->pid = pid;
    data->value = PAYLOAD_VALUE;
    data->unused = 0;
    KERMIT_SUBMIT(events, ctx, data);
  }
  u32 zero = 0;
  u64* count = completed.lookup(&zero);
  if (count)
    __sync_fetch_and_add(count, 1);
  return 0;
}
"""


class KermitTransportTests(unittest.TestCase):
    def load_probe(self, *, capture=False, attempts=1):
        self.pid = os.getpid()
        source = SOURCE.replace("TARGET_PID", str(self.pid)).replace(
            "RESERVE_ATTEMPTS", str(attempts)
        )
        bpf = (
            capture_bpf(source)
            if capture
            else compile_bpf(source, cflags=("-DKERMIT_QUEUE_SIZE=1",))
        )
        self.addCleanup(bpf.cleanup)
        bpf.attach_kprobe(event=bpf.get_syscall_fnname(b"getpid"), fn_name=b"emit")
        return bpf

    def test_capture_uses_existing_perf_consumer(self):
        bpf = self.load_probe(capture=True)
        records = []

        def on_event(cpu, data, size):
            event = bpf[b"events"].event(data)
            records.append(
                (cpu, event.pid, event.value, event.unused, ct.string_at(data, size))
            )

        bpf[b"events"].open_perf_buffer(on_event)
        os.getpid()
        bpf.perf_buffer_poll(timeout=100)
        self.assertTrue(records)
        cpu, pid, value, unused, raw = records[0]
        self.assertGreaterEqual(cpu, 0)
        self.assertEqual((pid, value, unused), (self.pid, 42, 0))
        self.assertEqual(raw[4:8], b"\0" * 4)
        self.assertEqual(raw[20:24], b"\0" * 4)
        with self.assertRaises(KeyError):
            bpf[b"mpmo_head_events"]

    def test_queue_publishes_in_mmap_and_reuses_unread_slot(self):
        bpf = self.load_probe()
        table = bpf[b"mpmo_buff_events"]
        with mmap.mmap(table.map_fd, mmap.PAGESIZE) as mapping:
            slot = table.Leaf.from_buffer(mapping)
            try:
                for initial_state in (0, 1):
                    with self.subTest(initial_state=initial_state):
                        slot.valid = initial_state
                        slot.data.value = 99
                        os.getpid()
                        self.assertEqual(slot.valid, 1)
                        self.assertEqual(slot.data.pid, self.pid)
                        self.assertEqual(slot.data.value, 42)
            finally:
                del slot
        self.assertGreaterEqual(bpf[b"mpmo_head_events"][ct.c_uint(0)].value, 2)

    def test_busy_slots_are_not_written_and_cleanup_still_runs(self):
        bpf = self.load_probe()
        table = bpf[b"mpmo_buff_events"]
        completed = bpf[b"completed"]
        with mmap.mmap(table.map_fd, mmap.PAGESIZE) as mapping:
            slot = table.Leaf.from_buffer(mapping)
            try:
                for initial_state in (2, 3):
                    with self.subTest(initial_state=initial_state):
                        slot.valid = initial_state
                        slot.data.value = 99
                        before = completed[ct.c_int(0)].value
                        os.getpid()
                        self.assertEqual(slot.valid, initial_state)
                        self.assertEqual(slot.data.value, 99)
                        self.assertGreater(completed[ct.c_int(0)].value, before)
            finally:
                del slot

    def test_zero_attempts_does_not_return_an_unclaimed_slot(self):
        bpf = self.load_probe(attempts=0)
        os.getpid()
        slot = bpf[b"mpmo_buff_events"][ct.c_uint(0)]
        self.assertEqual(slot.valid, 0)
        self.assertEqual(slot.data.value, 0)
        self.assertGreater(bpf[b"completed"][ct.c_int(0)].value, 0)


if __name__ == "__main__":
    unittest.main()

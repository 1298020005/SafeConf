# V2 live runtime progress

At19:11:41UTC, PID208033 had used22CPU seconds over311wall seconds and was waiting in folio_wait_bit_common. Only FULL_SUMS.npy/CELL_COUNTS.npy were open, and mapped writes had reached7,979,008,000bytes. Disk sampling showed25.2MiB/s writes,95.5%device utilization and179.9ms write latency on local ext4 /dev/sdb1. This confirms the initial phase was dominated by sum-map zeroing/writeback.

Initialization has since completed. The parent observed the process running at19:12UTC with a58,240,096byte tokens_0 spool; the latest independent sample also sees active first-shard decoding. The latest counter deltas since the first sample are 2,526,781,440additional write bytes and 4,837,376read bytes. File sizes and descriptor paths are recorded without reading expression values.

MemAvailable is111243500 kB, process RSS is722616 kB and peak RSS is731748 kB; /home initially had2.4TiB free. Low MemFree reflects a large filesystem cache. No RAM-cap increase is justified, and this audit changes no code, process or permit.

The continuation has live progress and remains within the8hour pipeline bound. Initialization precedes the hard600second per-file timer; first-shard decoding is now subject to that registered deadline. Full40-shard throughput is not yet measured, so completion is not guaranteed from this snapshot. Syscall/kernel-stack reads were denied and no escalation was attempted. No experiments or restarts were performed.

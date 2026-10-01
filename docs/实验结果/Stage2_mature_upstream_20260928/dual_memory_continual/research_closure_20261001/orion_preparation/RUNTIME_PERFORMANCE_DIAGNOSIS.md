# Aggregate runtime diagnosis

PID162145 had exited before sampling began. The registered log confirms exit1 at2026-10-01T18:52:38UTC after1497.943seconds, at the expression-loop per-file timeout (loader line385). HCT116_Batch1 completed; the failure occurred during the next fixed identity, HCT116_Batch10.

The600second timer resets for every file and is checked after generator yields and mapped-file flushes. Initialization and earlier files contribute to the nearly25minute total. Long decode/flush calls can defer this cooperative checkpoint; the code never promised a600second whole-job limit.

Inputs and outputs use local ext4 on /dev/sdb1 (/home), a rotational logical volume; NFS is absent. The sum map is28947×38606 float64 with8,940,223,056payload bytes and a fully allocated extent. The unfinished token spool is607,513,264bytes. Only shape, sizes and timestamps were inspected.

Live /proc CPU/I/O/page-fault counters were unavailable. The parent reported earlier61%CPU and D/R states; those are historical reports. Short post-exit samples showed0reads,0swap I/O and0%iowait and cannot attribute historical bottlenecks. No private expression values or cell distributions were opened, and no process was interrupted.

The confirmed failure is the per-file wall limit; CPU-versus-I/O shares remain unmeasured. Prioritize the vectorized decoder and sample its next actual run. If I/O remains material, consider flushing/dropping only dirty group-row ranges instead of the entire8.9GB mapping every512cells. A new exclusive file can also preserve zero-filled holes instead of eagerly writing zeros, after a targeted numerical/memory equivalence check. This audit implements neither change.

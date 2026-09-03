# On-call runbook

If retrieval latency exceeds the budget, check whether the index file was evicted from
the page cache after a deploy. A warm-up query loop restores it. Never disable the
permission filter to speed up a scan.

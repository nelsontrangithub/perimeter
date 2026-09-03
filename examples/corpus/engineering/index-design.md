# Index design notes

The retrieval index is a flat, memory-mapped array of binary and int8 codes. Access
control is applied inside the scan: the caller's permitted rows are gathered before any
distance is computed. Rescoring uses int8 codes against a float32 query. Contractors do
not have access to these notes.

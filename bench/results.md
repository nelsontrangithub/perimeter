| Metric | Measured | Budget |
|--------|---------:|-------:|
| p95 retrieval latency, all rows permitted | 5.70 ms | <= 30 ms |
| p95 retrieval latency, 10% of rows permitted | 3.09 ms | <= 30 ms |
| Peak RSS, sustained query loop | 248 MiB | <= 512 MiB |
| Index bytes per chunk | 1174 B | <= 1,280 B |
| recall@10 vs exact float32, all rows permitted | 0.976 | >= 0.95 |
| recall@10 vs exact float32, 10% permitted | 0.977 | >= 0.95 |
| Cohere API calls per query | 2 (1 embed, 1 rerank) | <= 2 |

Corpus: 50,000 chunks x 1024 dims (synthetic, clustered); k=10; 500 timed queries per caller after 50 warm-up; Cohere ports stubbed at zero cost (calls counted); p50 5.18 ms / p99 6.46 ms (all rows); ACL resolver calls per query 0.000 (cached); index build 3.3 s; on disk 56.0 MiB.

Environment: perimeter 0.1.0, python 3.12.13, numpy 2.5.2, platform Darwin arm64, cpu arm.

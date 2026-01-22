# RocksDB Test

## 目录
- [RocksDB Test](#rocksdb-test)
  - [目录](#目录)
  - [编译选项和测试选项](#编译选项和测试选项)
  - [运行性能测试](#运行性能测试)
    - [最简单的运行方法](#最简单的运行方法)
      - [使用 gcc10 运行](#使用-gcc10-运行)
      - [输出格式修改](#输出格式修改)


## 编译选项和测试选项

编译选项和模式对应关系
- Debug Mode：
    - 主要用于开发和测试阶段，帮助开发者发现和修复 bugs。
    - `make check`
    - `make all`
- Release Mode：
    - 适用于生产环境，注重性能和效率。
    - `make static_lib`
    - `make shared_lib`

其他选项
- PORTABLE选项：
    - `PORTABLE=1`：生成通用二进制文件，兼容性好但性能可能较低。
    - `PORTABLE=haswell`：针对特定CPU架构优化，性能更好但兼容性较低。
- DEBUG_LEVEL变量：
    - `DEBUG_LEVEL=1`：对应调试模式。
    - `DEBUG_LEVEL=0`：对应发布模式。

测试
- 单元测试：使用调试模式（`make check`），以便进行详细的错误检查和调试。
- 性能测试：使用发布模式（`make static_lib` 或 `make shared_lib`），以获得真实的性能数据。
- 压力测试：同样使用发布模式（`make static_lib` 或 `make shared_lib`），以模拟生产环境的负载条件。


运行所有的 unit test（耗时挺长的）：
``` bash
make clean
make check PORTABLE=1
```

压力测试：
db_stress 测试用于大规模验证数据的正确性
https://github.com/facebook/rocksdb/wiki/Stress-test
``` bash
make clean
DEBUG_LEVEL=1 make db_stress -j$(nproc)
./db_stress --db=/tmp/rocksdb_stress_test --ops_per_thread=1000000 --threads=32 --read_fault_one_in=1000
```




## 运行性能测试

### 最简单的运行方法

适用于 `rocksdb_test.json` 中最新的四个 commit

``` bash
make clean
make db_bench -j$(nproc) DEBUG_LEVEL=0
rm -rf /tmp/rocksdb_test
./db_bench --benchmarks=fillseq,readrandom --db=/tmp/rocksdb_test --num=1000000
```


在 commit 21db55f8164d2a6519dcc993f74bf7f49c700854 （2024-07-17T13:39:14-07:00）上的运行结果如下：

``` bash
zyw@pai-master:~/llm_on_code/rocksdb$ ./db_bench --benchmarks=fillseq,readrandom --db=/tmp/rocksdb_test --num=1000000
Set seed to 1736874265019867 because --seed was 0
Initializing RocksDB Options from the specified file
Initializing RocksDB Options from command-line flags
Integrated BlobDB: blob cache disabled
RocksDB:    version 9.5.0
Date:       Wed Jan 15 01:04:25 2025
CPU:        208 * Intel(R) Xeon(R) Platinum 8270 CPU @ 2.70GHz
CPUCache:   36608 KB
Keys:       16 bytes each (+ 0 bytes user-defined timestamp)
Values:     100 bytes each (50 bytes after compression)
Entries:    1000000
Prefix:    0 bytes
Keys per prefix:    0
RawSize:    110.6 MB (estimated)
FileSize:   62.9 MB (estimated)
Write rate: 0 bytes/second
Read rate: 0 ops/second
Compression: Snappy
Compression sampling rate: 0
Memtablerep: SkipListFactory
Perf Level: 1
------------------------------------------------
Initializing RocksDB Options from the specified file
Initializing RocksDB Options from command-line flags
Integrated BlobDB: blob cache disabled
DB path: [/tmp/rocksdb_test]
fillseq      :       1.521 micros/op 657338 ops/sec 1.521 seconds 1000000 operations;   72.7 MB/s
DB path: [/tmp/rocksdb_test]
readrandom   :       5.013 micros/op 199456 ops/sec 5.014 seconds 1000000 operations;   22.1 MB/s (1000000 of 1000000 found)
```


#### 使用 gcc10 运行

从 f9cfc6a808c9dc3ab7366edb10368559155d5172 （2022-07-06T09:30:25-07:00）开始及以前，需要使用 gcc10 来编译运行，否则会产生一箩筐的编译报错：

``` bash
make clean CC=gcc-10 CXX=g++-10
make db_bench -j$(nproc) DEBUG_LEVEL=0 CC=gcc-10 CXX=g++-10
rm -rf /tmp/rocksdb_test
./db_bench --benchmarks=fillseq,readrandom --db=/tmp/rocksdb_test --num=1000000
```



#### 输出格式修改

从 1ca1562e3565ac3d9ccfeeec2e206a21791f3aa3 （2022-03-21T17:30:51-07:00）开始及以前，输出的形式略有变化，最后四行如下：

``` bash
DB path: [/tmp/rocksdb_test]
fillseq      :       1.468 micros/op 681115 ops/sec;   75.3 MB/s
DB path: [/tmp/rocksdb_test]
readrandom   :       4.503 micros/op 222088 ops/sec;   24.6 MB/s (1000000 of 1000000 found)
```


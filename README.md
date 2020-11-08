# OneAgent NVBit Extension

## Foreword

Created by [Tomasz Gajger](https://github.com/tomix86).

**Notice**: although the author is a Dynatrace employee, this is a private project. It is not maintained nor endorsed by the Dynatrace.

The project is released under the [MIT License](LICENSE).

## Overview

A [Dynatrace OneAgent](https://www.dynatrace.com/support/help/) extension for gathering NVIDIA GPU metrics using [NVIDIA Binary Instrumentation Tool (NVBit)](https://github.com/NVlabs/NVBit).

The extension consists of two parts:

* [native module](nvbit-module/README.md), which is injected into monitored applications, gathers and publishes measurements,
* [Python extension](extension/README.md) responsible for providing configuration to the native module, retrieving the metrics, aggregating them and sending to Dynatrace cluster.

All metrics are process-specific and reported per-PGI. The extension is capable of monitoring multiple GPUs, the metrics coming from all the devices will be aggregated and sent as combined timeseries.
There is no support for sending separate timeseries per device.

### Requirements

* OneAgent version >= 1.191.
* [See Python part README](extension/README.md#requirements).
* [Compiled native module](nvbit-module/README.md#building).

## Setup and configuration

For a list of available configuration options, see [extension README](extension/README.md#configuration).

All processes for which the metrics should be gathered need to be [instrumented manually](nvbit-module/README.md#overview) with native module.

## Reported metrics

The table below outlines metrics collected by the extension. *Figures 1* and *2* exemplify how metrics are presented on the WebUI.

| Key                               | Metric description |
|-----------------------------------|--------------------|
| instructions_per_second           | Count of instuctions executed per second |
| gpu_occupancy                     | Average [occupancy](https://docs.nvidia.com/gameworks/index.html#developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm) achieved by kernels |
| gmem_access_coalescence           | Average [global memory accesses coalescence factor](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/) achieve by kernels |

If there are multiple GPUs present, the metrics will be displayed in a joint fashion, i.e:

* `instructions_per_second` will be a sum of instructions executed on all devices,
* `gpu_occupancy` and `gmem_access_coalescence` will be an average from per-device usage metrics.

![Host metrics display](docs/images/host_screen_keymetrics.png)
\
_Fig 1. Host metrics reported by the extension_

![PGI metrics display](docs/images/process_screen_metrics.png)
\
_Fig 2. PGI metrics reported by the extension_

## Planned metrics (future enhancements)

* Global memory access efficiency (coalescence factor)
* Branch divergence
* GPU time (the time it took for the computations on GPU to complete)

## Alerting

There are no built-in alerts defined yet.

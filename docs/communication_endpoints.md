# Communication endpoints specification

This document specifies communication between the _Python extension_ and _nvbit-module_.

## _nvbit-module_ runtime configuration

Runtime configuration is created on the fly by _Python extension_ and contains a list of process identifiers (pids) that should be instrumented, along with instrumentation functions to apply to each of them.
The file is saved in an atomic manner to ensure consistency.
_Python extension_ determines the list of pids internally based on `monitored_pg_names` specified in extension configuration.
The _nvbit-module_ polls the configuration file periodically to update its configuration.

### Config file format

Line-by-line format of the file:

```text
<pid>:<instrumentation_function_id>[,<instrumentation_function_id>...]
<pid_2>:<instrumentation_function_id>[,<instrumentation_function_id>...]
...
```

See [nvbit-module-runtime.conf](../nvbit-module/res/nvbit-module-runtime.conf) for an example.  

Instrumentation functions ids:

- `0` - instructions count
- `1` - occupancy
- `2` - global memory access coalescence factor

## Passing measurements from _nvbit-module_ to _Python extension_

Results are written to a dedicated directory, a file is created by every process publishing the measurements, in which the _nvbit-module_ instrumentation is active.
The files are created atomically to ensure consistency and named using the following pattern: `<pid>-<tid>-<timestamp>`.
_Python extension_ consumes measurement files and is responsible for their removal.
Aggregation is performed by _Python extension_.
Metrics are sent to the cluster only if _nvbit-module_ reported them for given entity.

The communication must be organized this direction (_nvbit-module_ to _Python extension_) to minimize the overhead incurred on monitored applications.
Specifically, the _nvbit-module_ cannot wait for the measurements to be received, otherwise it could halt the execution of instrumented application.
This issue could be resolved by spawning additional thread, but as the _nvbit-module_ is being run in the context of another process, doing so should be avoided. Additionally, it wouldn't help in scenarios where there are short-lived processes executing computations on the GPU as waiting for the results to be flushed, i.e. received by _Python extension_, would inevitably prolong the shutdown time of instrumented application.

In future, the protocol could be improved to make use of socket-based communication, e.g. HTTP over UDS or HTTP over local TCP socket.
Supporting RESTful interface would make the solution easier to re-use and extend.
However, this can only happen after custom extensions running continuously are supported in Dynatrace. At present, the extension is scheduled for execution every minute by the internal scheduling mechanism embedded into extensions runtime, and must terminate immediately after finishing its work.

### Measurements file format

Line-by-line format of the file:

```text
<instrumentation_primitive_id>:<measurement>
<instrumentation_primitive_id>:<measurement_2>
<instrumentation_primitive_id_2>:<measurement>
...
```

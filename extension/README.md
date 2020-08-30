# Extemsions implementation

## Overview

Note that the extension can attach metrics to multiple processes at once, but the metrics will only be displayed for processes that were specified in `processTypeNames` in `plugin.json`
If the process type is not specified there, then metrics will still be sent, but won't appear on the WebUI.
Currently there is no way to specify `Any` in `processTypeNames`, hence all the process types of interest need to be explicitly enumerated.

### Requirements

* For plugin development: [OneAgent Plugin SDK v1.191 or newer](https://dynatrace.github.io/plugin-sdk/index.html).
* Python >= 3.6.

## Configuration

* `enable_debug_log` - enables debug logging for troubleshooting purposes.
* `enable_intrumentation` - enables instrumentation
* `monitored_pg_names` - Comma-delimited list of process groups names that should be monitored
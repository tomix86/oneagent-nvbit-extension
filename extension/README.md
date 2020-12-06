# Extension implementation

## Overview

Python part of the extension, written using OneAgent Extensions SDK.

Note that the extension can send metrics for multiple processes at once, but the metrics will only be displayed for processes that were specified in `processTypeNames` in `plugin.json`.
If the process type is not specified there, then metrics will still be sent, but won't appear on the WebUI.
Currently there is no way to specify `Any` in `processTypeNames`, hence all the process types of interest need to be explicitly enumerated.

### Requirements

* For development: [OneAgent Extensions SDK v1.191 or newer](https://www.dynatrace.com/support/help/shortlink/extensions-hub#oneagent-extensions).
* Python >= 3.6.6.

## Configuration

* `enable_debug_log` - enables debug logging for troubleshooting purposes.
* `monitored_pg_names` - comma-delimited list of process groups names that should be monitored.
* `instrumentation_enabled` - enables instrumentation.
* `instrumentation_occupancy` - enables occupancy measurement.
* `instrumentation_code_injection` - (dropdown) enables selected code injection measurement.

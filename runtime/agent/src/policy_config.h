// The drafting policy's settings from a JSON file (runtime/controller/policies/*.json).
// Every key is optional; what is missing keeps the default of PolicyConfig.
//   {"bands": [{"max_depth": 1, "n_max": 16, "off_below": 0.10, "on_above": 0.12}, ...],
//    "window_drafted": 20000, "min_drafted": 400, "min_probe_drafted": 32, "probe_every": 4,
//    "probes_to_recover": 1, "error_cooldown": 20, "errors_to_latch": 3, "error_window": 200}
// Bands must be in rising order of max_depth. Throws std::runtime_error on a bad file.
#pragma once

#include "policy.h"
#include "tools.h"

namespace specedge {

PolicyConfig policy_from_json(const json & settings);
json policy_to_json(const PolicyConfig & config);

}  // namespace specedge

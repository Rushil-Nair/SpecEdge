#include "policy_config.h"

#include <stdexcept>

namespace specedge {

PolicyConfig policy_from_json(const json & settings) {
    if (!settings.is_object()) throw std::runtime_error("policy: expected a JSON object");
    PolicyConfig config;
    if (settings.contains("bands")) {
        config.bands.clear();
        int previous = 0;
        for (const auto & band : settings.at("bands")) {
            Band one{band.at("max_depth").get<int>(), band.at("n_max").get<int>(), band.at("off_below").get<double>(),
                     band.at("on_above").get<double>()};
            if (one.max_depth <= previous) throw std::runtime_error("policy: bands must be in rising order of max_depth");
            if (one.n_max < 0 || one.on_above < one.off_below) throw std::runtime_error("policy: a band needs n_max >= 0 and on_above >= off_below");
            previous = one.max_depth;
            config.bands.push_back(one);
        }
    }
    const auto number = [&settings](const char * key, int & into) {
        if (settings.contains(key)) into = settings.at(key).get<int>();
        if (into < 0) throw std::runtime_error(std::string("policy: ") + key + " must not be negative");
    };
    number("window_drafted", config.window_drafted);
    number("min_drafted", config.min_drafted);
    number("min_probe_drafted", config.min_probe_drafted);
    number("probe_every", config.probe_every);
    number("probes_to_recover", config.probes_to_recover);
    number("error_cooldown", config.error_cooldown);
    number("errors_to_latch", config.errors_to_latch);
    number("error_window", config.error_window);
    return config;
}

json policy_to_json(const PolicyConfig & config) {
    json bands = json::array();
    for (const auto & band : config.bands)
        bands.push_back(json{{"max_depth", band.max_depth}, {"n_max", band.n_max}, {"off_below", band.off_below}, {"on_above", band.on_above}});
    return json{{"bands", bands}, {"window_drafted", config.window_drafted}, {"min_drafted", config.min_drafted},
                {"min_probe_drafted", config.min_probe_drafted}, {"probe_every", config.probe_every},
                {"probes_to_recover", config.probes_to_recover}, {"error_cooldown", config.error_cooldown},
                {"errors_to_latch", config.errors_to_latch}, {"error_window", config.error_window}};
}

}  // namespace specedge

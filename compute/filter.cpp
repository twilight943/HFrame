#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

unordered_map<int, unordered_set<int>> dualsim_filter_qt(
    const vector<vector<int>>& query, 
    const vector<vector<int>>& data) {
    
    const vector<int>& data_vid = data[0];
    const vector<int>& data_vlbl = data[1];
    const vector<int>& data_srcid = data[2];
    const vector<int>& data_dstid = data[3];
    
    const vector<int>& query_vid = query[0];
    const vector<int>& query_vlbl = query[1];
    const vector<int>& query_srcid = query[2];
    const vector<int>& query_dstid = query[3];
    
    unordered_map<int, vector<int>> data_out_edges;
    unordered_map<int, vector<int>> data_in_edges;
    unordered_map<int, int> data_labels;
    
    for (size_t i = 0; i < data_srcid.size(); i++) {
        int src = data_srcid[i];
        int dst = data_dstid[i];
        data_out_edges[src].push_back(dst);
        data_in_edges[dst].push_back(src);
    }
    
    for (size_t i = 0; i < data_vid.size(); i++) {
        data_labels[data_vid[i]] = data_vlbl[i];
    }
    
    unordered_map<int, vector<int>> query_out_edges;
    unordered_map<int, vector<int>> query_in_edges;
    unordered_map<int, int> query_labels;
    
    for (size_t i = 0; i < query_srcid.size(); i++) {
        int src = query_srcid[i];
        int dst = query_dstid[i];
        query_out_edges[src].push_back(dst);
        query_in_edges[dst].push_back(src);
    }
    
    for (size_t i = 0; i < query_vid.size(); i++) {
        query_labels[query_vid[i]] = query_vlbl[i];
    }
    
    unordered_map<int, unordered_set<int>> sim;
    for (const auto& u : query_vid) {
        sim[u] = unordered_set<int>();
        int u_label = query_labels[u];
        
        for (const auto& v : data_vid) {
            if (data_labels[v] == u_label) {
                sim[u].insert(v);
            }
        }
        
        if (sim[u].empty()) {
            return {};
        }
    }
    
    bool changed;
    int maxloop = 0;
    int nowloop = 0;
    do {
        changed = false;
        if (maxloop > 0 && nowloop == maxloop) {
            break;
        } else {
            nowloop++;
        }
        for (const auto& u : query_vid) {
            auto& sim_u = sim[u];
            auto sim_u_copy = sim_u;
            
            for (int v : sim_u_copy) {
                bool has_match = true;
                
                for (int u_prime : query_out_edges[u]) {
                    const auto& sim_u_prime = sim[u_prime];
                    bool found_edge = false;
                    
                    for (int v_prime : data_out_edges[v]) {
                        if (sim_u_prime.count(v_prime)) {
                            found_edge = true;
                            break;
                        }
                    }
                    
                    if (!found_edge) {
                        has_match = false;
                        break;
                    }
                }
                
                if (!has_match) {
                    sim_u.erase(v);
                    changed = true;
                    
                    if (sim_u.empty()) {
                        return {};
                    }
                }
            }
        }
        
        for (const auto& u : query_vid) {
            auto& sim_u = sim[u];
            auto sim_u_copy = sim_u;
            
            for (int v : sim_u_copy) {
                bool has_match = true;
                
                for (int u_prime : query_in_edges[u]) {
                    const auto& sim_u_prime = sim[u_prime];
                    bool found_edge = false;
                    
                    for (int v_prime : data_in_edges[v]) {
                        if (sim_u_prime.count(v_prime)) {
                            found_edge = true;
                            break;
                        }
                    }
                    
                    if (!found_edge) {
                        has_match = false;
                        break;
                    }
                }
                
                if (!has_match) {
                    sim_u.erase(v);
                    changed = true;
                    
                    if (sim_u.empty()) {
                        return {};
                    }
                }
            }
        }
        
    } while (changed);
    
    return sim;
}

PYBIND11_MODULE(filter, m) {
    m.doc() = "pybind11 example plugin";
    m.def("dualsim_filter_qt", &dualsim_filter_qt, "dualsim_filter_qt");
}
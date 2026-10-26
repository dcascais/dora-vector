// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "vector/node_topology.h"

#include "dora/vector/status.h"

#include <errno.h>
#include <stdint.h>

int dora_node_topology_link(struct dora_node_topology* nodes, uint32_t node_count) {
    if (nodes == nullptr || node_count == 0 || nodes[0].parent_node_index != UINT32_MAX) {
        return EINVAL;
    }
    for (uint32_t node = 0; node < node_count; ++node) {
        nodes[node].first_child_index = UINT32_MAX;
        nodes[node].next_sibling_index = UINT32_MAX;
        if (node != 0 && nodes[node].parent_node_index >= node) {
            return EINVAL;
        }
    }
    for (uint32_t cursor = node_count; cursor > 1;) {
        const uint32_t node = --cursor;
        const uint32_t parent_node_index = nodes[node].parent_node_index;
        nodes[node].next_sibling_index = nodes[parent_node_index].first_child_index;
        nodes[parent_node_index].first_child_index = node;
    }
    return DORA_OK;
}

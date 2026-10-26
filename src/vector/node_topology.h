// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Preorder vector topology.

#ifndef DORA_VECTOR_NODE_TOPOLOGY_H
#define DORA_VECTOR_NODE_TOPOLOGY_H

#include "dora/vector/view.h"
#include "utf8/input.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Preorder Node Topology
//
//------------------------------------------------------------------------------

// Requires preorder; node 0 has parent_node_index == UINT32_MAX.
struct dora_node_topology {
    struct dora_vector_view view;
    enum dora_utf8_proof utf8_proof;
    uint32_t parent_node_index;
    uint32_t child_index;
    uint32_t first_child_index;
    uint32_t next_sibling_index;
};

int dora_node_topology_link(struct dora_node_topology* nodes, uint32_t node_count);

#endif

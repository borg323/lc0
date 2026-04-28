/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lczero {
namespace classic {

class Node;

// Per-node search-local state in the search overlay.
// Holds ephemeral data that belongs to the current search session rather
// than the persistent game tree stored in Node.
struct SearchTreeNode {
  // Virtual loss / in-flight counter for this node.
  // Tracks how many search threads are currently visiting this node.
  uint32_t n_in_flight = 0;
};

// A lightweight search-local overlay keyed by Node*.
//
// SearchTree owns search-local bookkeeping that should not live in the
// persistent Node tree, including:
//   - per-node virtual-loss (n_in_flight) accounting,
//   - shared collision path bookkeeping.
//
// The persistent Node tree continues to own long-lived state: edges, n_,
// wl_, d_, m_, bounds, and terminal flags.
//
// All methods that mutate SearchTree state must be called while the caller
// holds the search nodes_mutex_ (write lock), matching the existing
// invariant for Node::n_in_flight_.
class SearchTree {
 public:
  explicit SearchTree(Node* root);

  // -----------------------------------------------------------------------
  // n_in_flight / virtual-loss operations.
  // All callers must hold nodes_mutex_ (write lock).
  // -----------------------------------------------------------------------

  // Returns the current n_in_flight for @node, or 0 if not tracked.
  uint32_t GetNInFlight(const Node* node) const;

  // Returns node->GetN() + GetNInFlight(node).
  int GetNStarted(const Node* node) const;

  // If the node is "being extended" (n==0 && n_in_flight>0) return false.
  // Otherwise increment n_in_flight and return true.
  bool TryStartScoreUpdate(Node* node);

  // Decrement n_in_flight by @multivisit (cancels a pending visit).
  void CancelScoreUpdate(Node* node, int multivisit);

  // Increment n_in_flight by @multivisit.
  void IncrementNInFlight(Node* node, int multivisit);

  // Called when a score update is finalized: decrements n_in_flight by
  // @multivisit.  (The corresponding n_ increment is still done by
  // Node::FinalizeScoreUpdate.)
  void FinalizeScoreUpdate(Node* node, int multivisit);

  // -----------------------------------------------------------------------
  // Shared collision bookkeeping.
  // -----------------------------------------------------------------------

  // Record a new shared collision (path + multivisit count).
  void AddSharedCollision(std::vector<Node*> path, int multivisit);

  // Cancel all pending shared collisions: decrement n_in_flight along each
  // stored path and clear the list.
  void CancelSharedCollisions();

 private:
  // Returns (creating if necessary) the per-node overlay entry.
  SearchTreeNode& GetOrCreate(Node* node);

  // Returns a pointer to the per-node overlay entry, or nullptr if absent.
  const SearchTreeNode* GetIfExists(const Node* node) const;

  Node* root_;
  std::unordered_map<Node*, SearchTreeNode> nodes_;

  // Pending shared collision paths with their multivisit counts.
  // Each entry is a (path, multivisit) pair where path[0] is the root.
  std::vector<std::pair<std::vector<Node*>, int>> shared_collisions_;
};

}  // namespace classic
}  // namespace lczero

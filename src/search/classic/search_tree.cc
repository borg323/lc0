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

#include "search/classic/search_tree.h"

#include "search/classic/node.h"

namespace lczero {
namespace classic {

SearchTree::SearchTree(Node* root) : root_(root) {}

SearchTreeNode& SearchTree::GetOrCreate(Node* node) {
  return nodes_[node];
}

const SearchTreeNode* SearchTree::GetIfExists(const Node* node) const {
  auto it = nodes_.find(const_cast<Node*>(node));
  if (it == nodes_.end()) return nullptr;
  return &it->second;
}

uint32_t SearchTree::GetNInFlight(const Node* node) const {
  const auto* stn = GetIfExists(node);
  return stn ? stn->n_in_flight : 0;
}

int SearchTree::GetNStarted(const Node* node) const {
  return static_cast<int>(node->GetN()) +
         static_cast<int>(GetNInFlight(node));
}

bool SearchTree::TryStartScoreUpdate(Node* node) {
  auto& stn = GetOrCreate(node);
  if (node->GetN() == 0 && stn.n_in_flight > 0) return false;
  ++stn.n_in_flight;
  return true;
}

void SearchTree::CancelScoreUpdate(Node* node, int multivisit) {
  GetOrCreate(node).n_in_flight -= multivisit;
}

void SearchTree::IncrementNInFlight(Node* node, int multivisit) {
  GetOrCreate(node).n_in_flight += multivisit;
}

void SearchTree::FinalizeScoreUpdate(Node* node, int multivisit) {
  GetOrCreate(node).n_in_flight -= multivisit;
}

void SearchTree::AddSharedCollision(std::vector<Node*> path, int multivisit) {
  shared_collisions_.emplace_back(std::move(path), multivisit);
}

void SearchTree::CancelSharedCollisions() {
  for (auto& entry : shared_collisions_) {
    auto& path = entry.first;
    // Skip the leaf node (path.back()); cancel from its ancestors up to root.
    for (auto it = ++(path.crbegin()); it != path.crend(); ++it) {
      CancelScoreUpdate(*it, entry.second);
    }
  }
  shared_collisions_.clear();
}

}  // namespace classic
}  // namespace lczero

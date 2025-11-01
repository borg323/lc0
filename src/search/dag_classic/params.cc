/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2025 The LCZero Authors

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

#include "search/dag_classic/params.h"

namespace lczero {
namespace dag_classic {

const OptionId SearchParams::kUseUncertaintyWeightingId{
    {.long_flag = "use-uncertainty-weighting",
     .uci_option = "UseUncertaintyWeighting",
     .help_text = "Enable uncertainty weighting in MCTS."}};
const OptionId SearchParams::kUncertaintyWeightingCapId{
    {.long_flag = "uncertainty-weighting-cap",
     .uci_option = "UncertaintyWeightingCap",
     .help_text = "Cap for node weight from uncertainty weighting."}};
const OptionId SearchParams::kUncertaintyWeightingCoefficientId{
    {.long_flag = "uncertainty-weighting-coefficient",
     .uci_option = "UncertaintyWeightingCoefficient",
     .help_text = "Coefficient in the uncertainty weighting formula."}};
const OptionId SearchParams::kUncertaintyWeightingExponentId{
    {.long_flag = "uncertainty-weighting-exponent",
     .uci_option = "UncertaintyWeightingExponent",
     .help_text = "Exponent in the uncertainty weighting formula."}};

void SearchParams::Populate(OptionsParser* options) {
  BaseSearchParams::Populate(options);
  options->Add<BoolOption>(kUseUncertaintyWeightingId) = false;
  options->Add<FloatOption>(kUncertaintyWeightingCapId, 0.0f, 10.0f) = 1.0f;
  options->Add<FloatOption>(kUncertaintyWeightingCoefficientId, 0.0f, 10.0f) =
      0.13f;
  options->Add<FloatOption>(kUncertaintyWeightingExponentId, -10.0f, 0.0f) =
      -1.76f;
}

SearchParams::SearchParams(const OptionsDict& options)
    : BaseSearchParams(options) {}
}  // namespace dag_classic
}  // namespace lczero

/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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

#include "chess/pgn.h"

#include <gtest/gtest.h>

namespace lczero {

namespace {
std::vector<Opening> Parse(const std::string& pgn) {
  PgnReader reader;
  reader.AddPgnString(pgn);
  return reader.ReleaseGames();
}

std::string Moves(const Opening& o) {
  std::string res;
  for (const auto& m : o.moves) res += m.ToString(false) + " ";
  if (!res.empty()) res.pop_back();
  return res;
}
}  // namespace

TEST(Pgn, Basic) {
  auto g = Parse("[Event \"x\"]\n[Result \"1-0\"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n");
  ASSERT_EQ(g.size(), 1u);
  EXPECT_EQ(g[0].start_fen, ChessBoard::kStartposFen);
  EXPECT_EQ(Moves(g[0]), "e2e4 e7e5 g1f3 b8c6");
}

TEST(Pgn, CommentsVariationsNagsAndAnnotations) {
  auto g = Parse(
      "[Event \"multi\nline\"]\n[Site \"]\"]\n\n"
      "1.e4! {comment [with] stuff\n over lines} e5?! $1 (1... c5 2. Nf3 "
      "(2. Nc3 d6) 2... d6) 2. Nf3 ; rest of line 1. d4\n"
      "% escape line\n"
      "Nc6+ 3. Bb5 *\n");
  ASSERT_EQ(g.size(), 1u);
  EXPECT_EQ(Moves(g[0]), "e2e4 e7e5 g1f3 b8c6 f1b5");
}

TEST(Pgn, MultipleGamesNoBlankLinesAndMissingResult) {
  auto g = Parse(
      "[Event \"a\"]\n1. d4 d5\n[Event \"b\"]\n1. c4 e6 1/2-1/2\n"
      "\xEF\xBB\xBF[Event \"c\"]\n\n*\n");
  ASSERT_EQ(g.size(), 3u);
  EXPECT_EQ(Moves(g[0]), "d2d4 d7d5");
  EXPECT_EQ(Moves(g[1]), "c2c4 e7e6");
  EXPECT_EQ(Moves(g[2]), "");
}

TEST(Pgn, CastlingPromotionEnPassant) {
  auto g = Parse(
      "[FEN \"r3k2r/1P6/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1\"]\n\n"
      "1. exd6 O-O 2. O-O-O Rfd8 3. bxa8=Q+ Kg7 1-0\n");
  ASSERT_EQ(g.size(), 1u);
  EXPECT_EQ(g[0].start_fen, "r3k2r/1P6/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1");
  EXPECT_EQ(Moves(g[0]), "e5d6 e8g8 e1c1 f8d8 b7a8q g8g7");
}

TEST(Pgn, BlackToMoveFen) {
  auto g = Parse("[FEN \"4k3/8/8/8/8/8/4p3/K7 b - - 0 1\"]\n\n1... e1=Q# 0-1");
  EXPECT_EQ(g.size(), 1u);
}

TEST(Pgn, Illegal) {
  EXPECT_THROW(Parse("1. e5 *"), Exception);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}

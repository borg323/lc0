/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

#include <zlib.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {

struct Opening {
  std::string start_fen = ChessBoard::kStartposFen;
  MoveList moves;
};

inline bool GzGetLine(gzFile file, std::string& line) {
  bool flag = false;
  char s[2000];
  line.clear();
  while (gzgets(file, s, sizeof(s))) {
    flag = true;
    line += s;
    auto r = line.find_last_of('\n');
    if (r != std::string::npos) {
      line.erase(r);
      break;
    }
  }
  return flag;
}

// Recursive descent parser for PGN (PGN standard, section 8 "Import format"
// with the usual leniency). Only the main line of each game is kept; recursive
// annotation variations, comments, NAGs, escape lines and tags (other than
// FEN) are parsed and discarded.
//
// Grammar:
//   file       := game*
//   game       := tag_pair* movetext
//   tag_pair   := '[' symbol string ... ']'
//   movetext   := (move_number | san | nag | variation | result)*
//   variation  := '(' movetext ')'
class PgnReader {
 public:
  void AddPgnFile(const std::string& filepath) {
    const gzFile file = gzopen(filepath.c_str(), "r");
    if (!file) {
      throw Exception(errno == ENOENT ? "Opening book file not found."
                                      : "Error opening opening book file.");
    }
    std::string data;
    char buf[1 << 16];
    int n;
    while ((n = gzread(file, buf, sizeof(buf))) > 0) data.append(buf, n);
    const bool failed = n < 0;
    gzclose(file);
    if (failed) throw Exception("Error reading opening book file.");
    AddPgnString(data);
  }

  void AddPgnString(std::string data) {
    data_ = std::move(data);
    pos_ = 0;
    has_token_ = false;
    // Ignore a UTF-8 BOM.
    if (data_.compare(0, 3, "\xEF\xBB\xBF") == 0) pos_ = 3;
    while (ParseGame()) {
    }
    data_.clear();
  }

  std::vector<Opening> GetGames() const { return games_; }
  std::vector<Opening>&& ReleaseGames() { return std::move(games_); }

 private:
  enum class TokenType {
    kEof,
    kString,
    kSymbol,
    kPeriod,
    kNag,
    kLBracket,
    kRBracket,
    kLParen,
    kRParen,
    kAsterisk,
    kOther,
  };

  struct Token {
    TokenType type = TokenType::kEof;
    std::string text;
  };

  // ---- Lexer ----

  static bool IsSymbolStart(unsigned char c) {
    return std::isalnum(c) || c == '!' || c == '?' || c == '-' || c == '@' ||
           c >= 0x80;
  }
  static bool IsSymbolChar(unsigned char c) {
    return std::isalnum(c) || c == '_' || c == '+' || c == '#' || c == '=' ||
           c == ':' || c == '-' || c == '/' || c == '!' || c == '?' ||
           c == '@' || c >= 0x80;
  }

  Token LexToken() {
    while (pos_ < data_.size()) {
      const unsigned char c = data_[pos_];
      if (std::isspace(c)) {
        ++pos_;
      } else if (data_.compare(pos_, 3, "\xEF\xBB\xBF") == 0) {
        pos_ += 3;  // UTF-8 BOM, possibly from concatenated files.
      } else if (c == '%' && (pos_ == 0 || data_[pos_ - 1] == '\n')) {
        SkipLine();
      } else if (c == ';') {
        SkipLine();
      } else if (c == '{') {
        const auto end = data_.find('}', pos_);
        pos_ = end == std::string::npos ? data_.size() : end + 1;
      } else {
        break;
      }
    }
    if (pos_ >= data_.size()) return {};
    const unsigned char c = data_[pos_];
    switch (c) {
      case '"': {
        Token t{TokenType::kString, {}};
        ++pos_;
        while (pos_ < data_.size() && data_[pos_] != '"') {
          if (data_[pos_] == '\\' && pos_ + 1 < data_.size()) ++pos_;
          t.text += data_[pos_++];
        }
        if (pos_ < data_.size()) ++pos_;
        return t;
      }
      case '[': ++pos_; return {TokenType::kLBracket, "["};
      case ']': ++pos_; return {TokenType::kRBracket, "]"};
      case '(': ++pos_; return {TokenType::kLParen, "("};
      case ')': ++pos_; return {TokenType::kRParen, ")"};
      case '.': ++pos_; return {TokenType::kPeriod, "."};
      case '*': ++pos_; return {TokenType::kAsterisk, "*"};
      case '$': {
        Token t{TokenType::kNag, {}};
        ++pos_;
        while (pos_ < data_.size() && std::isdigit(
                   static_cast<unsigned char>(data_[pos_]))) {
          t.text += data_[pos_++];
        }
        return t;
      }
      default:
        break;
    }
    if (IsSymbolStart(c)) {
      Token t{TokenType::kSymbol, {}};
      while (pos_ < data_.size() &&
             IsSymbolChar(static_cast<unsigned char>(data_[pos_]))) {
        t.text += data_[pos_++];
      }
      // Old style en passant suffix "e.p." glued to the previous move.
      if (t.text == "e" && data_.compare(pos_, 3, ".p.") == 0) {
        pos_ += 3;
        return {TokenType::kOther, "e.p."};
      }
      return t;
    }
    ++pos_;
    return {TokenType::kOther, std::string(1, static_cast<char>(c))};
  }

  void SkipLine() {
    const auto end = data_.find('\n', pos_);
    pos_ = end == std::string::npos ? data_.size() : end + 1;
  }

  const Token& Peek() {
    if (!has_token_) {
      token_ = LexToken();
      has_token_ = true;
    }
    return token_;
  }

  Token Next() {
    Peek();
    has_token_ = false;
    return std::move(token_);
  }

  // ---- Parser ----

  static bool IEquals(const std::string& a, const char* b) {
    return a.size() == std::strlen(b) &&
           std::equal(a.begin(), a.end(), b, [](unsigned char x, char y) {
             return std::toupper(x) == std::toupper(static_cast<unsigned char>(y));
           });
  }

  static bool IsResult(const std::string& s) {
    return s == "1-0" || s == "0-1" || s == "1/2-1/2" || s == "1/2" ||
           s.compare(0, 2, "\xC2\xBD") == 0;
  }

  // Parses one game. Returns false when there is nothing left to parse.
  bool ParseGame() {
    if (Peek().type == TokenType::kEof) return false;
    ++game_number_;
    bool seen_anything = false;
    cur_startpos_ = ChessBoard::kStartposFen;
    cur_board_.SetFromFen(cur_startpos_);
    cur_game_.clear();
    while (Peek().type == TokenType::kLBracket) {
      ParseTag();
      seen_anything = true;
    }
    seen_anything |= ParseMovetext(true);
    if (seen_anything) games_.push_back({cur_startpos_, cur_game_});
    return true;
  }

  void ParseTag() {
    Next();  // '['
    if (Peek().type != TokenType::kSymbol) {
      SkipToTagEnd();
      return;
    }
    const std::string name = Next().text;
    std::string value;
    if (Peek().type == TokenType::kString) value = Next().text;
    SkipToTagEnd();
    if (IEquals(name, "FEN")) {
      cur_startpos_ = value;
      try {
        cur_board_.SetFromFen(cur_startpos_);
      } catch (const Exception& e) {
        throw Exception("PGN game " + std::to_string(game_number_) +
                        ": bad FEN tag: " + e.what());
      }
    }
  }

  void SkipToTagEnd() {
    while (Peek().type != TokenType::kEof &&
           Peek().type != TokenType::kLBracket) {
      if (Next().type == TokenType::kRBracket) return;
    }
  }

  // Parses movetext until the game termination (main line) or the closing
  // parenthesis (variation). Returns true if any token was consumed.
  bool ParseMovetext(bool main_line) {
    bool consumed = false;
    while (true) {
      switch (Peek().type) {
        case TokenType::kEof:
        case TokenType::kLBracket:  // Start of next game (missing result).
          return consumed;
        case TokenType::kRParen:
          if (!main_line) return consumed;
          Next();  // Stray parenthesis.
          break;
        case TokenType::kLParen:
          ParseVariation();
          break;
        case TokenType::kAsterisk:
          Next();
          if (main_line) return true;
          break;
        case TokenType::kSymbol: {
          const std::string sym = Next().text;
          if (IsResult(sym)) {
            if (main_line) return true;
          } else if (main_line) {
            ParseSymbol(sym);
          }
          break;
        }
        default:
          Next();
          break;
      }
      consumed = true;
    }
  }

  void ParseVariation() {
    Next();  // '('
    ParseMovetext(false);
    if (Peek().type == TokenType::kRParen) Next();
  }

  // Handles a symbol in main line: move number, annotation or SAN move.
  void ParseSymbol(const std::string& sym) {
    const unsigned char c = sym[0];
    if (c == '!' || c == '?') return;
    if (std::all_of(sym.begin(), sym.end(),
                    [](unsigned char x) { return std::isdigit(x); })) {
      return;  // Move number.
    }
    if (c >= 0x80) return;
    if (std::isdigit(c) && sym.compare(0, 3, "0-0") != 0) return;
    if (sym == "--" || sym == "Z0" || sym == "@@@@") {
      throw Exception("PGN game " + std::to_string(game_number_) +
                      ": null moves are not supported.");
    }
    Move m;
    try {
      m = SanToMove(sym, cur_board_);
    } catch (const Exception& e) {
      throw Exception("PGN game " + std::to_string(game_number_) + ", move '" +
                      sym + "': " + e.what());
    }
    const bool black = cur_board_.flipped();
    cur_board_.ApplyMove(m);
    // Board ApplyMove wants mirrored for black, but outside code wants
    // normal, so mirror it back again.
    if (black) m.Flip();
    cur_game_.push_back(m);
    cur_board_.Mirror();
  }

  // ---- SAN ----

  static std::optional<PieceType> PromotionPiece(char c) {
    switch (std::toupper(static_cast<unsigned char>(c))) {
      case 'Q': return kQueen;
      case 'R': return kRook;
      case 'B': return kBishop;
      case 'N': return kKnight;
      default: return std::nullopt;
    }
  }

  // Returns a move in the board's own (side to move is white) frame.
  static Move SanToMove(std::string san, const ChessBoard& board) {
    // Strip check/mate/annotation suffixes and en passant marker.
    while (!san.empty() && std::strchr("+#!?", san.back())) san.pop_back();
    if (san.empty()) throw Exception("Empty move.");
    const auto plm = board.GenerateLegalMoves();

    if (san[0] == 'O' || san[0] == '0') {
      if (san != "O-O" && san != "O-O-O" && san != "0-0" && san != "0-0-0") {
        throw Exception("Malformed castling move.");
      }
      const bool queenside = san.size() == 5;
      const auto king_board = board.kings() & board.ours();
      const Square king_sq(
          File::FromIdx(GetLowestBit(king_board.as_int())), kRank1);
      const Move m = Move::WhiteCastling(
          king_sq.file(), queenside ? board.castlings().our_queenside_rook
                                    : board.castlings().our_kingside_rook);
      if (std::find(plm.begin(), plm.end(), m) == plm.end()) {
        throw Exception("Illegal castling move.");
      }
      return m;
    }

    size_t idx = 0;
    BitBoard pieces = board.pawns() & board.ours();
    switch (san[0]) {
      case 'K': pieces = board.kings() & board.ours(); idx = 1; break;
      case 'Q': pieces = board.queens() & board.ours(); idx = 1; break;
      case 'R': pieces = board.rooks() & board.ours(); idx = 1; break;
      case 'B': pieces = board.bishops() & board.ours(); idx = 1; break;
      case 'N': pieces = board.knights() & board.ours(); idx = 1; break;
      default: break;
    }
    const bool is_pawn = idx == 0;

    // Remove capture markers and promotion '='.
    std::string core;
    bool capture = false;
    for (; idx < san.size(); ++idx) {
      if (san[idx] == 'x' || san[idx] == ':') {
        capture = true;
      } else if (san[idx] != '=' && san[idx] != '-' && san[idx] != '(' &&
                 san[idx] != ')') {
        core += san[idx];
      }
    }
    // Last rank digit marks the destination square.
    const auto j = core.find_last_of("12345678");
    if (j == std::string::npos || j < 1 || core[j - 1] < 'a' ||
        core[j - 1] > 'h') {
      throw Exception("Malformed move.");
    }
    std::optional<PieceType> promotion;
    if (j + 1 < core.size()) {
      promotion = PromotionPiece(core[j + 1]);
      if (!promotion || j + 2 != core.size()) {
        throw Exception("Malformed promotion.");
      }
    }
    const std::string from_spec = core.substr(0, j - 1);
    int from_file = -1;
    int from_rank = -1;
    for (char c : from_spec) {
      if (c >= 'a' && c <= 'h' && from_file < 0) {
        from_file = c - 'a';
      } else if (c >= '1' && c <= '8' && from_rank < 0) {
        from_rank = c - '1';
      } else {
        throw Exception("Malformed move.");
      }
    }
    // Move frame is mirrored for black.
    if (board.flipped() && from_rank >= 0) from_rank = 7 - from_rank;
    int to_rank = core[j] - '1';
    if (board.flipped()) to_rank = 7 - to_rank;
    const Square to(File::FromIdx(core[j - 1] - 'a'), Rank::FromIdx(to_rank));

    std::optional<Move> found;
    for (const Move& m : plm) {
      if (m.is_castling() || m.to() != to) continue;
      const Square from = m.from();
      if (!pieces.get(from)) continue;
      if (from_file >= 0 && from.file().idx != from_file) continue;
      if (from_rank >= 0 && from.rank().idx != from_rank) continue;
      // A pawn push stays on its file; captures name the from file.
      if (is_pawn && !capture && from_file < 0 && from.file().idx != to.file().idx) {
        continue;
      }
      if (m.is_promotion() != promotion.has_value()) continue;
      if (promotion && m.promotion() != *promotion) continue;
      if (found) throw Exception("Move is ambiguous.");
      found = m;
    }
    if (!found) throw Exception("Move is illegal.");
    return *found;
  }

  std::string data_;
  size_t pos_ = 0;
  Token token_;
  bool has_token_ = false;
  int game_number_ = 0;

  ChessBoard cur_board_{ChessBoard::kStartposFen};
  MoveList cur_game_;
  std::string cur_startpos_ = ChessBoard::kStartposFen;
  std::vector<Opening> games_;
};

}  // namespace lczero

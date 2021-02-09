/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020-2021 The LCZero Authors

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

#include "utils/numa.h"

#include "chess/bitboard.h"
#include "utils/logging.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace lczero {

std::map<int, Numa::Group> Numa::groups = {};
int Numa::thread_count = 0;
int Numa::core_count = 0;

void Numa::Init(int x) {
#if defined(_WIN64) && _WIN32_WINNT >= 0x0601
  int group_count = 0;
  thread_count = 0;
  core_count = 0;

  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* buffer;
  DWORD len;
  GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
  buffer = static_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(malloc(len));
  GetLogicalProcessorInformationEx(RelationProcessorCore, buffer, &len);
  for (int offset = 0; offset < len;) {
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* info =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((char*)buffer + offset);
    if (info->Processor.EfficiencyClass == x) {
      int group = info->Processor.GroupMask[0].Group;
      if (groups.find(group) == groups.end()) {
        Group tmp = {};
        groups.emplace(group, tmp);
        group_count++;
      }
      int threads = BitBoard(info->Processor.GroupMask[0].Mask).count();
      thread_count += threads;
      core_count++;
      groups[group].cores++;
      groups[group].threads += threads;
      groups[group].mask |= info->Processor.GroupMask[0].Mask;
    }
    offset += info->Size;
  }
  free(buffer);

  CERR << "Detected " << core_count << " core(s) and " << thread_count
       << " thread(s) in " << group_count << " group(s).";

  for (int group_id = 0; group_id < group_count; group_id++) {
    int group_threads = groups[group_id].threads;
    int group_cores = groups[group_id].cores;
    CERR << "Group " << group_id << " has " << group_cores << " core(s) and "
         << group_threads << " thread(s).";
  }
#else
  // Silence warning.
  (void)x;
#endif
}

void Numa::BindThread(int id) {
#if defined(_WIN64) && _WIN32_WINNT >= 0x0601
  int group_count = groups.size();

  int core_id = id;
  GROUP_AFFINITY affinity = {};
  for (int group_id = 0; group_id < group_count; group_id++) {
    int group_threads = groups[group_id].threads;
    int group_cores = groups[group_id].cores;
    // Allocate cores of each group in order, and distribute remaining threads
    // to all groups.
    if ((id < core_count && core_id < group_cores) ||
        (id >= core_count && (id - core_count) % group_count == group_id)) {
      affinity.Group = group_id;
      affinity.Mask = groups[group_id].mask;
      SetThreadGroupAffinity(GetCurrentThread(), &affinity, NULL);
      break;
    }
    core_id -= group_cores;
  }
#else
  // Silence warning.
  (void)id;
#endif
}

}  // namespace lczero

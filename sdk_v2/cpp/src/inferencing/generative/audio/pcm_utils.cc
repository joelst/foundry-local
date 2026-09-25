// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inferencing/generative/audio/pcm_utils.h"

#include <cstring>
#include <fstream>

namespace fl {

std::vector<float> ConvertS16LEToFloat(const uint8_t* pcm_bytes, size_t byte_count) {
  const size_t sample_count = byte_count / 2;
  std::vector<float> samples(sample_count);

  for (size_t i = 0; i < sample_count; ++i) {
    // Little-endian: low byte first, high byte second.
    int16_t sample;
    std::memcpy(&sample, pcm_bytes + i * 2, sizeof(int16_t));
    samples[i] = static_cast<float>(sample) / 32768.0f;
  }

  return samples;
}

std::optional<double> AudioInternal::TryReadWavDurationSeconds(const std::string& audio_file_path) {
  std::ifstream in(audio_file_path, std::ios::binary);
  char riff[4];
  uint32_t riff_size = 0;
  char wave[4];
  in.read(riff, sizeof(riff));
  in.read(reinterpret_cast<char*>(&riff_size), sizeof(riff_size));
  in.read(wave, sizeof(wave));
  if (!in || std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(wave, "WAVE", 4) != 0) {
    return std::nullopt;
  }

  uint32_t byte_rate = 0;
  while (in) {
    char chunk_id[4];
    uint32_t chunk_size = 0;
    in.read(chunk_id, sizeof(chunk_id));
    in.read(reinterpret_cast<char*>(&chunk_size), sizeof(chunk_size));
    if (!in) {
      return std::nullopt;
    }

    if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
      if (chunk_size < 16) {
        return std::nullopt;
      }

      // Skip audio format (2), channels (2) and sample rate (4) to reach the byte rate.
      in.seekg(8, std::ios::cur);
      in.read(reinterpret_cast<char*>(&byte_rate), sizeof(byte_rate));
      in.seekg(static_cast<std::streamoff>(chunk_size) - 12 + (chunk_size % 2), std::ios::cur);
    } else if (std::strncmp(chunk_id, "data", 4) == 0) {
      if (byte_rate == 0) {
        return std::nullopt;
      }

      return static_cast<double>(chunk_size) / byte_rate;
    } else {
      in.seekg(static_cast<std::streamoff>(chunk_size) + (chunk_size % 2), std::ios::cur);
    }
  }

  return std::nullopt;
}

}  // namespace fl

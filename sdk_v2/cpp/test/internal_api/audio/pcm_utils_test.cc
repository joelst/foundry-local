// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Unit tests for PCM s16le → float32 conversion.
//

#include "inferencing/generative/audio/pcm_utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace fl;

namespace {

void AppendU32(std::string& out, uint32_t v) {
  out.append(reinterpret_cast<const char*>(&v), sizeof(v));
}

void AppendU16(std::string& out, uint16_t v) {
  out.append(reinterpret_cast<const char*>(&v), sizeof(v));
}

// Minimal RIFF/WAVE file; an optional odd-sized LIST chunk before "data" exercises chunk skipping and padding.
std::string MakeWav(uint32_t sample_rate, uint16_t channels, uint32_t data_bytes, bool with_list_chunk) {
  std::string body = "WAVE";
  body += "fmt ";
  AppendU32(body, 16);
  AppendU16(body, 1);
  AppendU16(body, channels);
  AppendU32(body, sample_rate);
  AppendU32(body, sample_rate * channels * 2);
  AppendU16(body, static_cast<uint16_t>(channels * 2));
  AppendU16(body, 16);
  if (with_list_chunk) {
    body += "LIST";
    AppendU32(body, 3);
    body += "abc";
    body += '\0';
  }

  body += "data";
  AppendU32(body, data_bytes);
  body.append(data_bytes, '\0');

  std::string wav = "RIFF";
  AppendU32(wav, static_cast<uint32_t>(body.size()));
  return wav + body;
}

class TempFile {
 public:
  explicit TempFile(const std::string& contents)
      : path_(std::filesystem::temp_directory_path() /
              ("pcm_utils_test_" + std::to_string(reinterpret_cast<uintptr_t>(this)) + ".wav")) {
    std::ofstream(path_, std::ios::binary) << contents;
  }

  ~TempFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  std::string path() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

}  // namespace

TEST(PcmUtilsTest, Silence_AllZeros) {
  std::vector<uint8_t> pcm(64, 0);
  auto result = ConvertS16LEToFloat(pcm.data(), pcm.size());

  EXPECT_EQ(result.size(), 32u);
  for (float sample : result) {
    EXPECT_FLOAT_EQ(sample, 0.0f);
  }
}

TEST(PcmUtilsTest, KnownValues) {
  // 0x0100 in LE = 256 as int16_t → 256/32768 ≈ 0.0078125
  uint8_t pcm[] = {0x00, 0x01};
  auto result = ConvertS16LEToFloat(pcm, sizeof(pcm));

  ASSERT_EQ(result.size(), 1u);
  EXPECT_FLOAT_EQ(result[0], 256.0f / 32768.0f);
}

TEST(PcmUtilsTest, MaxPositive) {
  // INT16_MAX = 32767 = 0xFF7F in LE
  uint8_t pcm[] = {0xFF, 0x7F};
  auto result = ConvertS16LEToFloat(pcm, sizeof(pcm));

  ASSERT_EQ(result.size(), 1u);
  EXPECT_NEAR(result[0], 1.0f, 0.001f);
}

TEST(PcmUtilsTest, MaxNegative) {
  // INT16_MIN = -32768 = 0x0080 in LE
  uint8_t pcm[] = {0x00, 0x80};
  auto result = ConvertS16LEToFloat(pcm, sizeof(pcm));

  ASSERT_EQ(result.size(), 1u);
  EXPECT_FLOAT_EQ(result[0], -1.0f);
}

TEST(PcmUtilsTest, OddByteCount_TrailingByteIgnored) {
  // 3 bytes → only 1 sample (first 2 bytes), trailing byte ignored
  uint8_t pcm[] = {0x00, 0x01, 0xFF};
  auto result = ConvertS16LEToFloat(pcm, sizeof(pcm));

  ASSERT_EQ(result.size(), 1u);
  EXPECT_FLOAT_EQ(result[0], 256.0f / 32768.0f);
}

TEST(PcmUtilsTest, EmptyInput) {
  auto result = ConvertS16LEToFloat(nullptr, 0);
  EXPECT_TRUE(result.empty());
}

TEST(PcmUtilsTest, MultipleSamples) {
  // Two samples: 0 and -1 (0xFFFF = -1 in int16_t)
  uint8_t pcm[] = {0x00, 0x00, 0xFF, 0xFF};
  auto result = ConvertS16LEToFloat(pcm, sizeof(pcm));

  ASSERT_EQ(result.size(), 2u);
  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_FLOAT_EQ(result[1], -1.0f / 32768.0f);
}

TEST(PcmUtilsTest, WavDurationFromHeader) {
  TempFile mono16k(MakeWav(16000, 1, 16000 * 2 * 3, false));
  auto duration = AudioInternal::TryReadWavDurationSeconds(mono16k.path());
  ASSERT_TRUE(duration.has_value());
  EXPECT_DOUBLE_EQ(*duration, 3.0);
}

TEST(PcmUtilsTest, WavDurationSkipsExtraChunksAndIgnoresSampleRate) {
  TempFile stereo44k(MakeWav(44100, 2, 44100 * 4 / 2, true));
  auto duration = AudioInternal::TryReadWavDurationSeconds(stereo44k.path());
  ASSERT_TRUE(duration.has_value());
  EXPECT_DOUBLE_EQ(*duration, 0.5);
}

TEST(PcmUtilsTest, WavDurationUnknownForNonWavOrMissingFile) {
  TempFile mp3("ID3 not a wav file");
  EXPECT_FALSE(AudioInternal::TryReadWavDurationSeconds(mp3.path()).has_value());
  EXPECT_FALSE(AudioInternal::TryReadWavDurationSeconds("does_not_exist_pcm_utils_test.wav").has_value());

  TempFile truncated(MakeWav(16000, 1, 32, false).substr(0, 30));
  EXPECT_FALSE(AudioInternal::TryReadWavDurationSeconds(truncated.path()).has_value());
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <optional>
#include <span>

namespace fl::AudioInternal {

/// Model-specific Whisper control token IDs. IDs differ between Whisper variants (e.g. `<|0.00|>` is 50364 in
/// tiny and 50365 in large-v3-turbo), so they must be resolved from each model's tokenizer rather than hard-coded.
struct WhisperTimestampTokens {
  int32_t eot = 0;              // <|endoftext|>
  int32_t no_timestamps = 0;    // <|notimestamps|>
  int32_t timestamp_begin = 0;  // <|0.00|>; every ID >= this is a timestamp in 0.02s steps
};

/// Whisper's default `max_initial_timestamp` of 1.0s expressed in 0.02s timestamp steps.
inline constexpr int kWhisperMaxInitialTimestampIndex = 50;

/// Apply OpenAI Whisper's timestamp decoding rules (ApplyTimestampRules in openai/whisper decoding.py) to one row of
/// next-token logits, in place. Without these rules, greedy decoding picks `<|notimestamps|>` as the first token even
/// when the prompt omits it, so no timestamp tokens are ever generated.
///
/// Rules: suppress `<|notimestamps|>` and other non-EOT control tokens; timestamps appear in pairs (except before
/// EOT); timestamps never decrease and every segment has non-zero length; the first token must be a timestamp no later
/// than `max_initial_timestamp_index`; and a timestamp is forced whenever the total timestamp probability exceeds the
/// most likely text token.
///
/// @param logits     Next-token logits for a single sequence (length = vocab size).
/// @param generated  Tokens generated so far, excluding the prompt.
/// Leaves `logits` untouched if the token IDs are inconsistent with the vocabulary size.
void ApplyWhisperTimestampRules(std::span<float> logits,
                                std::span<const int32_t> generated,
                                const WhisperTimestampTokens& tokens,
                                std::optional<int> max_initial_timestamp_index = kWhisperMaxInitialTimestampIndex);

}  // namespace fl::AudioInternal

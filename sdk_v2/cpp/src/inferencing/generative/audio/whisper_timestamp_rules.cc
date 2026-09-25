// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "inferencing/generative/audio/whisper_timestamp_rules.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace fl::AudioInternal {

namespace {

constexpr float kMasked = -std::numeric_limits<float>::infinity();

void Mask(std::span<float> logits, size_t begin, size_t end) {
  end = std::min(end, logits.size());
  if (begin < end) {
    std::fill(logits.begin() + static_cast<std::ptrdiff_t>(begin), logits.begin() + static_cast<std::ptrdiff_t>(end),
              kMasked);
  }
}

// log(sum(exp(x))) over the unmasked entries; -inf when every entry is masked.
float LogSumExp(std::span<const float> values) {
  float max_value = kMasked;
  for (float v : values) {
    max_value = std::max(max_value, v);
  }

  if (!std::isfinite(max_value)) {
    return kMasked;
  }

  double sum = 0.0;
  for (float v : values) {
    sum += std::exp(static_cast<double>(v) - max_value);
  }

  return static_cast<float>(std::log(sum) + max_value);
}

}  // namespace

void ApplyWhisperTimestampRules(std::span<float> logits,
                                std::span<const int32_t> generated,
                                const WhisperTimestampTokens& tokens,
                                std::optional<int> max_initial_timestamp_index) {
  const size_t vocab = logits.size();
  if (tokens.eot < 0 || tokens.timestamp_begin <= tokens.eot || static_cast<size_t>(tokens.timestamp_begin) >= vocab ||
      tokens.no_timestamps <= tokens.eot || tokens.no_timestamps >= tokens.timestamp_begin) {
    return;
  }

  const auto eot = static_cast<size_t>(tokens.eot);
  const auto ts_begin = static_cast<size_t>(tokens.timestamp_begin);

  // Control tokens between <|endoftext|> and <|0.00|> (<|notimestamps|>, <|startoftranscript|>, language/task
  // tokens, ...) are never valid transcription output in timestamp mode.
  Mask(logits, eot + 1, ts_begin);

  const size_t n = generated.size();
  const bool last_was_timestamp = n >= 1 && generated[n - 1] >= tokens.timestamp_begin;
  const bool penultimate_was_timestamp = n < 2 || generated[n - 2] >= tokens.timestamp_begin;

  if (last_was_timestamp) {
    if (penultimate_was_timestamp) {
      Mask(logits, ts_begin, vocab);  // a closing+opening pair was just emitted: text must follow
    } else {
      Mask(logits, 0, eot);  // an unpaired timestamp must be followed by another timestamp or EOT
    }
  }

  auto last_timestamp = std::find_if(generated.rbegin(), generated.rend(),
                                     [&](int32_t t) { return t >= tokens.timestamp_begin; });
  if (last_timestamp != generated.rend()) {
    // Timestamps must not decrease; a segment's closing timestamp must also be strictly after its opening one.
    const auto last = static_cast<size_t>(*last_timestamp);
    const size_t limit = (last_was_timestamp && !penultimate_was_timestamp) ? last : last + 1;
    Mask(logits, ts_begin, limit);
  }

  if (n == 0) {
    Mask(logits, 0, ts_begin);
    if (max_initial_timestamp_index.has_value() && *max_initial_timestamp_index >= 0) {
      Mask(logits, ts_begin + static_cast<size_t>(*max_initial_timestamp_index) + 1, vocab);
    }
  }

  // log_softmax subtracts the same normalizer from every entry, so comparing raw-logit logsumexp against the max
  // text logit is equivalent to the reference's log-probability comparison.
  const std::span<const float> text_logits(logits.data(), ts_begin);
  const std::span<const float> timestamp_logits(logits.data() + ts_begin, vocab - ts_begin);
  const float timestamp_logsumexp = LogSumExp(timestamp_logits);
  const float max_text_logit = *std::max_element(text_logits.begin(), text_logits.end());
  if (timestamp_logsumexp > max_text_logit) {
    Mask(logits, 0, ts_begin);
  }
}

}  // namespace fl::AudioInternal

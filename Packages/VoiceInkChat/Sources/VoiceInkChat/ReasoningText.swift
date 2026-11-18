import Foundation

/// Removes reasoning that some models write inline in their answer text.
///
/// Open-weight reasoning models served through OpenAI-compatible APIs, Bedrock, or Ollama
/// (for models that don't support `think`) can return their reasoning inside the answer,
/// wrapped in `<think>…</think>`. Some chat templates open the tag in the prompt, so only the
/// closing tag appears in the output.
enum ReasoningText {
    static func strip(_ text: String) -> String {
        let open = "<think>"
        let close = "</think>"
        var result = text

        // An answer that starts inside an open block: everything up to the first lone close tag.
        if let closeRange = result.range(of: close, options: .caseInsensitive) {
            let before = result[..<closeRange.lowerBound]
            if before.range(of: open, options: .caseInsensitive) == nil {
                result = String(result[closeRange.upperBound...])
            }
        }

        // Complete blocks anywhere in the answer.
        while let openRange = result.range(of: open, options: .caseInsensitive) {
            if let closeRange = result.range(
                of: close,
                options: .caseInsensitive,
                range: openRange.upperBound..<result.endIndex
            ) {
                result.removeSubrange(openRange.lowerBound..<closeRange.upperBound)
            } else if result[..<openRange.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                // A block that's opened at the start and never closed: the model ran out of
                // tokens while reasoning, so there's no answer.
                result = ""
            } else {
                // An unclosed tag after real text is more likely part of the answer.
                break
            }
        }
        return result
    }
}

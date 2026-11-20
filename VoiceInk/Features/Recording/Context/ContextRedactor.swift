import Foundation

/// Masks secrets and personal numbers in captured context (window OCR, clipboard) before it
/// reaches an AI provider. Context only has to show what the user is talking about, so the rules
/// go by shape: long numbers and random-looking tokens go, words, names and paths stay.
enum ContextRedactor {
    static let mask = "•••"

    static func redact(_ text: String) -> String {
        rules.reduce(text) { text, rule in
            rule.regex.stringByReplacingMatches(
                in: text,
                range: NSRange(text.startIndex..., in: text),
                withTemplate: rule.template
            )
        }
    }

    private struct Rule {
        let regex: NSRegularExpression
        let template: String

        /// Masks the whole match, or everything after capture group 1 when `keepsLabel` is set.
        init(_ pattern: String, keepsLabel: Bool = false) {
            // The patterns are constants, so a bad one is a programming error.
            regex = try! NSRegularExpression(pattern: pattern)
            template = keepsLabel ? "$1\(ContextRedactor.mask)" : ContextRedactor.mask
        }
    }

    private static let rules: [Rule] = [
        // PEM private keys. Without an END line (cut off on screen), mask the base64 lines that follow.
        Rule(
            #"-{2,9}\s*BEGIN[A-Z ]*PRIVATE KEY\s*-{2,9}(?:[\s\S]*?-{2,9}\s*END[A-Z ]*PRIVATE KEY\s*-{2,9}|(?:\s*[A-Za-z0-9+/=\\]{16,})*)"#
        ),
        // Database URLs, and any URL with a password in it.
        Rule(
            #"(?i)\b(?:(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|rediss?|amqps?|mssql|sqlserver|jdbc:[a-z]+)://|[a-z][a-z0-9+.-]{0,20}://[^\s:/@"'<>]*:[^\s/@"'<>&]*@)[^\s"'<>]*"#
        ),
        // A value with a digit after a secret-sounding name: DB_PASS=…, "apiKey": "…", API Key: …
        // Paths and code stay: values starting like a path, calls like `token = uuid4()`, and
        // anything without a digit, like `token = getToken()`.
        Rule(
            #"(?i)((?:passw(?:or)?d|passphrase|pwd|[_-]pass|[_-]pw|secret|token|credentials?|(?:(?:api|access|private|secret)[ _-]?|[_-])key)(?:[_-][A-Za-z0-9]+)*["']?[ \t]*[:=](?!=)[ \t]*["']?)(?![~.]{0,2}/)(?![^\s"',;]*\()(?=[^\s"',;]*\d)[^\s"',;]+"#,
            keepsLabel: true
        ),
        // API keys, tokens and hashes: 20+ characters in which letters and digits switch places 3+
        // times. Random strings shuffle them; names keep digits in one or two spots
        // (claude-haiku-4-5-20251001, ISO8601DateFormatter). Anything in a word with a / is a path.
        Rule(
            #"(?<![A-Za-z0-9_+/-])(?=(?:[A-Za-z0-9_+-]*?(?:[A-Za-z](?=[0-9])|[0-9](?=[A-Za-z]))){3})[A-Za-z0-9_+-]{20,}(?![A-Za-z0-9_+/-])(?<!/\S{0,200})"#
        ),
        // SSNs (123-45-6789, 123 45 6789), card numbers (1234 5678 9012 3456), and any run of
        // 9+ digits standing on their own: SSNs and card numbers without separators, account numbers.
        // Digits glued to letters, or in a word with a /, are part of a name, URL or path.
        Rule(#"(?<![A-Za-z0-9_/])(?:\d{3}([- ])\d{2}\1\d{4}|\d{4}([- ])\d{4}\2\d{4}\2\d{4}|\d{9,})(?![A-Za-z0-9_/])(?<!/\S{0,200})"#),
    ]
}

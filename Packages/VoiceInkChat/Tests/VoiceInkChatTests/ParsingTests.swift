import Foundation
import Testing
@testable import VoiceInkChat

@Suite struct ServerSentEventParserTests {
    @Test func handlesEveryLineEndingAndSplitCRLF() {
        var parser = ServerSentEventParser()
        var events = parser.feed("data: one\r")
        events += parser.feed("\n\r\ndata: two\rdata: more\r\r")
        events += parser.feed("id: 7\nevent: ping\ndata\n\n")
        #expect(events == [
            ServerSentEvent(data: "one"),
            ServerSentEvent(data: "two\nmore"),
            ServerSentEvent(id: "7", event: "ping", data: ""),
        ])
    }

    @Test func skipsCommentsBOMAndEventsWithoutData() {
        var parser = ServerSentEventParser()
        let events = parser.feed("\u{FEFF}: keep-alive\n\nevent: empty\n\ndata:no-space\n\n")
        #expect(events == [ServerSentEvent(data: "no-space")])
    }

    @Test func joinsUTF8SplitAcrossChunks() {
        var parser = ServerSentEventParser()
        let bytes = Array("data: café ☕️\n\n".utf8)
        var events: [ServerSentEvent] = []
        for byte in bytes { events += parser.feed(Data([byte])) }
        #expect(events == [ServerSentEvent(data: "café ☕️")])
    }

    @Test func finishFlushesALastEventWithoutBlankLine() {
        var parser = ServerSentEventParser()
        #expect(parser.feed("data: [DONE]").isEmpty)
        #expect(parser.finish() == [ServerSentEvent(data: "[DONE]")])
    }
}

@Suite struct ReasoningTextTests {
    @Test func stripsBlocksAndKeepsAnswers() {
        #expect(ReasoningText.strip("<think>a</think>Answer") == "Answer")
        #expect(ReasoningText.strip("<THINK>a</THINK>\n\nAnswer") == "\n\nAnswer")
        #expect(ReasoningText.strip("<thinking>x</thinking>A<reasoning>y</reasoning>B") == "AB")
        #expect(ReasoningText.strip("reasoning only</think>Answer") == "Answer")
        #expect(ReasoningText.strip("<think>never closed") == "")
        #expect(ReasoningText.strip("Use a <think> tag in HTML") == "Use a <think> tag in HTML")
        #expect(ReasoningText.strip("Plain answer.") == "Plain answer.")
    }
}

@Suite struct JSONValueTests {
    @Test func roundTripsAndEncodesWholeNumbersAsIntegers() throws {
        let value: JSONValue = ["a": 1, "b": 0.5, "c": .null, "d": [true, "x"], "e": ["f": 1024]]
        let encoded = String(data: try value.encoded(), encoding: .utf8)!
        #expect(encoded == #"{"a":1,"b":0.5,"c":null,"d":[true,"x"],"e":{"f":1024}}"#)
        #expect(try JSONValue(data: Data(encoded.utf8)) == value)
    }
}

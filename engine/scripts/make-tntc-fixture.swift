// Writes TNTC chunks for the Rust compatibility test.
//
// The script is compiled together with the engine's own chunk code, so the
// format, the header and the magic are the real ones rather than a copy:
//
//   swiftc -parse-as-library -O -o /tmp/make-tntc-fixture \
//     engine/Sources/TintaEngine/Protocol.swift \
//     engine/Sources/TintaEngine/ChunkStore.swift \
//     engine/scripts/make-tntc-fixture.swift
//   /tmp/make-tntc-fixture <outdir>
//
// The output has one track, `audio/mic`, with two chunks and a gap between them:
//   chunk 0: start 0,    1000 samples, sample i = i * 3 - 1500
//   chunk 1: start 3000,  500 samples, sample i = 2000 - i * 2
// The key is the 32 bytes 0x00..0x1f, so the fixture is deterministic.
//
// `ChunkWriter` would round the samples through Float, so this seals Int16
// samples directly with `ChunkFormat.header` and a fixed nonce.

import CryptoKit
import Foundation

@main
struct TntcFixture {
    static let key = Data((0..<32).map { UInt8($0) })

    static func makeChunk(startSample: Int64, samples: [Int16], nonceByte: UInt8) throws -> Data {
        let nonce = try AES.GCM.Nonce(data: Data(repeating: nonceByte, count: 12))
        let pcm = samples.withUnsafeBufferPointer { Data(buffer: $0) }
        let sealed = try AES.GCM.seal(pcm, using: SymmetricKey(data: key), nonce: nonce)
        guard let combined = sealed.combined else {
            throw NSError(
                domain: "tntc-fixture", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "seal failed"])
        }
        var file = ChunkFormat.header(startSample: startSample, sampleCount: samples.count)
        file.append(combined)
        return file
    }

    static func main() throws {
        guard CommandLine.arguments.count == 2 else {
            FileHandle.standardError.write(Data("usage: make-tntc-fixture <outdir>\n".utf8))
            exit(2)
        }
        let outdir = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        let directory = outdir.appendingPathComponent("audio/mic", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let first = (0..<1000).map { Int16($0 * 3 - 1500) }
        let second = (0..<500).map { Int16(2000 - $0 * 2) }
        try makeChunk(startSample: 0, samples: first, nonceByte: 0)
            .write(to: directory.appendingPathComponent("00000000.tntc"))
        try makeChunk(startSample: 3000, samples: second, nonceByte: 1)
            .write(to: directory.appendingPathComponent("00000001.tntc"))
        print("wrote \(directory.path)")
    }
}

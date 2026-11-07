// Writes TNTC chunks with CryptoKit for the Rust compatibility test.
//
//   swift engine/scripts/make-tntc-fixture.swift <outdir>
//
// The output has one track, `audio/mic`, with two chunks and a gap between them:
//   chunk 0: start 0,    1000 samples, sample i = i * 3 - 1500
//   chunk 1: start 3000,  500 samples, sample i = 2000 - i * 2
// The key is the 32 bytes 0x00..0x1f, so the fixture is deterministic.

import CryptoKit
import Foundation

let key = Data((0..<32).map { UInt8($0) })

func makeChunk(startSample: Int64, samples: [Int16], nonceByte: UInt8) throws -> Data {
    let nonce = try AES.GCM.Nonce(data: Data(repeating: nonceByte, count: 12))
    let pcm = samples.withUnsafeBufferPointer { Data(buffer: $0) }
    let sealed = try AES.GCM.seal(pcm, using: SymmetricKey(data: key), nonce: nonce)
    guard let combined = sealed.combined else {
        throw NSError(domain: "tntc-fixture", code: 1, userInfo: [NSLocalizedDescriptionKey: "seal failed"])
    }
    var file = Data("TNTC".utf8)
    var start = startSample.littleEndian
    var count = UInt32(samples.count).littleEndian
    file.append(Data(bytes: &start, count: 8))
    file.append(Data(bytes: &count, count: 4))
    file.append(combined)
    return file
}

func main() throws {
    guard CommandLine.arguments.count == 2 else {
        FileHandle.standardError.write(Data("usage: swift make-tntc-fixture.swift <outdir>\n".utf8))
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

do {
    try main()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}

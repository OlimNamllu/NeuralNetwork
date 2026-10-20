//
//  main.swift
//  NeuralNetwork
//
//  Created by Milo Ullman on 3/8/26.
//


import Foundation

// MARK: - Saving Parameters with Data()
@discardableResult
func writeParameters(_ params: [Double], to filename: String) -> Bool {
    let data = params.withUnsafeBufferPointer { Data(buffer: $0) }
    let url = URL(filePath: getPath(filename: filename))
    do {
        try data.write(to: url, options: .atomic)
        print("wrote \(params.count) parameters to \(filename)")
        return true
    } catch {
        print("could not write \(filename): \(error)")
        return false
    }
}

func readParameters(from filename: String) -> [Double]? {
    let url = URL(filePath: getPath(filename: filename))

    guard let data = try? Data(contentsOf: url) else {
        print("no parameter file at \(filename)")
        return nil
    }

    guard data.count % MemoryLayout<Double>.stride == 0 else {
        print("\(filename) is \(data.count) bytes, not a multiple of \(MemoryLayout<Double>.stride)")
        return nil
    }

    let count = data.count / MemoryLayout<Double>.stride
    let params = data.withUnsafeBytes { raw in
        (0..<count).map { i in
            raw.loadUnaligned(fromByteOffset: i * MemoryLayout<Double>.stride, as: Double.self)
        }
    }
    print("read \(params.count) parameters from \(filename)")
    return params
}

// MARK: - Saving Scores JSON

var scores: [Double] = []

let path2 = "/Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/Scores.swift"
let url2 = URL(fileURLWithPath: path2)

func writeScores(_ scores: [Double], title: String) {
    let swift = "let scores\(title): [Double] = \(scores) \n \n"
    
    if let fileHandle = try? FileHandle(forWritingTo: url2) {
        defer { try? fileHandle.close() } // Always close the file stream
        let _ = try? fileHandle.seekToEnd()
        if let data = swift.data(using: .utf8) {
            try? fileHandle.write(contentsOf: data)
        }
    }
}

var tim = "/Users/tim/Documents/neural_network/NeuralNetwork/"
var milo = "/Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/"

func getPath(filename: String) -> String {
    return tim + filename
}


// MARK: - Files

// Files for saving parameters between training runs
let parametersBefore = "Parameters-before.bin"
let parametersAfter = "Parameters-after.bin"


// 36000 iterations, 0.03-0.05 noise, 28 character context, 756 256 180 27 network layers
let iter36k = "iter36k.bin"

// Building off previous network, 9k more iterations at 0.07 noise
let iter45k = "iter42k.bin"

// 18000 iterations 486, 256, 180, 96, 27 Network, higher noise
let iter18k5L = "iter18k5L.bin"

// 18000 iterations 486, 256, 180, 96, 27 Network, higher noise
let iter30k5L = "iter30k5L.bin"

let trainingImages = fetchCGImagesFromLocalPath(folderPath: "/Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/pumpkin")


// MARK: - EXECUTE
var network = NeuralNetwork()


let sample = trainingImages[200]

var datum = ImageDatum(cgimage: sample)
ImageDatum.exportCGImageToJpeg(img: datum.getCGImage()!, name: "saving", overrideName: "file:///Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/original.jpg")

let inputs = datum.getTraningDatum().0


// PRETRAINING
var pre = network.run(datum.getTraningDatum().0).0.map { UInt8(abs($0) * 255) }
var rgbpretrain: [ImageDatum.RGBPixel] = []
for i in 0..<(pre.count / 3) {
    let step = i * 3
    var pixel = ImageDatum.RGBPixel(r: 0, g: 0, b: 0)
    pixel.r = pre[step + 0]
    pixel.g = pre[step + 1]
    pixel.b = pre[step + 2]
    
    rgbpretrain.append(pixel)
}
let pretrain = ImageDatum(from: rgbpretrain)
ImageDatum.exportCGImageToJpeg(img: pretrain.getCGImage()!, name: "saving", overrideName: "file:///Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/pretrain.jpg")

print("\n\n PRINTING POST TRAINING  OUTPUTS:")
print(pre)



// TRAINING
print("Training...")
network.backprop(iterations: 5, learnRate: 0.003, batchSize: 15)
print("Training Complete")


// POST TRAINING
var post = network.run(datum.getTraningDatum().0).0.map { UInt8(abs($0) * 255) }
var rgbposttrain: [ImageDatum.RGBPixel] = []
for i in 0..<(post.count / 3) {
    let step = i * 3
    var pixel = ImageDatum.RGBPixel(r: 0, g: 0, b: 0)
    pixel.r = post[step + 0]
    pixel.g = post[step + 1]
    pixel.b = post[step + 2]
    rgbposttrain.append(pixel)
}
let posttrain = ImageDatum(from: rgbposttrain)
ImageDatum.exportCGImageToJpeg(img: posttrain.getCGImage()!, name: "saving", overrideName: "file:///Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/posttrain.jpg")


print("\n\n PRINTING POST TRAINING  OUTPUTS:")
print(post)

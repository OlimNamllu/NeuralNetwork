//
//  ImageProcessing.swift
//  NeuralNetwork
//
//  Created by Milo Ullman on 17/10/26.
//
import CoreGraphics
import ImageIO
import Foundation

let file = "pumpkin"
let url = URL(filePath: getPath(filename: file))

func loadCGImagesFromBundle(folderName: String) -> [CGImage] {
    let fileManager = FileManager.default
    guard let bundleURL = Bundle.main.resourceURL?.appendingPathComponent(folderName) else { return [] }
    
    do {
        let fileURLs = try fileManager.contentsOfDirectory(at: bundleURL, includingPropertiesForKeys: nil)
        let imageURLs = fileURLs.filter { ["jpg"].contains($0.pathExtension.lowercased()) }
        
        return imageURLs.compactMap { url -> CGImage? in
            guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
            return CGImageSourceCreateImageAtIndex(imageSource, 0, nil)
        }
    } catch {
        print("Error loading CGImages: \(error)")
        return []
    }
}

struct ImageDatum {
    struct RGBPixel {
        var r: UInt8
        var g: UInt8
        var b: UInt8
    }
    
    struct BWPixel {
        var c: UInt8
    }
    
    var RGB: [RGBPixel]
    var BW: [BWPixel]
    
    var width: Int = 52
    var height: Int = 64
    
    
    init(cgimage: CGImage) {
        guard let dataProvider = cgimage.dataProvider,
              let data = dataProvider.data,
              let pointer = CFDataGetBytePtr(data) else {
            fatalError("Couldn't access image data")
        }
        
        self.RGB = []
        self.BW = []

        let width = cgimage.width
        let height = cgimage.height
        let bytesPerRow = cgimage.bytesPerRow
        let bytesPerPixel = cgimage.bitsPerPixel / cgimage.bitsPerComponent
        
        self.width = width
        self.height = height

        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * bytesPerRow) + (x * bytesPerPixel)
                let red = pointer[offset]
                let green = pointer[offset + 1]
                let blue = pointer[offset + 2]
                self.RGB.append(RGBPixel(r: red, g: green, b: blue))
            }
        }
        
        for pixel in self.RGB {
            let grayscale = (pixel.r / 3) + (pixel.g / 3) + (pixel.b / 3)
            self.BW.append(BWPixel(c: grayscale))
        }
    }
    
}

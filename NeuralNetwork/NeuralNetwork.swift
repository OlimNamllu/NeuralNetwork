import Foundation

// MARK: - Network

struct CapturedNeurons {
    var inputNeurons: [CGFloat]
    var outputNeurons: [CGFloat]
    var hiddenLayers: [[CGFloat]]
    
    init() {
        self.inputNeurons = []
        self.outputNeurons = []
        self.hiddenLayers = []
    }
}

class NeuralNetwork {
    // static let layers = [textContextSize * 27, 312, 27]
    
    static let layers = [2, 3, 4, 2]
    
    struct HiddenLayer: Equatable {
        var biases: [CGFloat]
        
        //This is the weight to the next layer not previous
        var weights: [[CGFloat]]
        
        init(layerCount: Int, nextLayerCount: Int) {
            biases = (0..<layerCount).map { _ in
                CGFloat.random(in: -0.3...0.3)
            }
            
            weights = (0..<layerCount).map { _ in
                (0..<nextLayerCount).map { _ in
                    CGFloat.random(in: -0.3...0.3)
                }
            }
        }
    }
    
    
    var hiddenLayers: [HiddenLayer] = {
        var newLayers = layers
        newLayers.removeFirst()
        
        var outputSize = newLayers.removeLast()
        
        return newLayers.enumerated().map { (index, layerSize) in
            if index < newLayers.count - 1 {
                return HiddenLayer(layerCount: layerSize, nextLayerCount: newLayers[index + 1])
            } else {
                return HiddenLayer(layerCount: layerSize, nextLayerCount: outputSize)
            }
        }
    }()
    
    var inputWeights: [[CGFloat]] = (0..<layers[0]).map { _ in
        (0..<layers[1]).map { _ in CGFloat.random(in: -0.3...0.3) } }
    
    var outputBiases: [CGFloat] = (0..<layers.last!).map { _ in CGFloat.random(in: -0.1...0.1) }
    
    
    struct Metrics {
        var mse: CGFloat
        var accuracy: CGFloat
        var top3: CGFloat
        
        func printOut() -> String {
            return "MSE: \(mse), ACCURACY: \(accuracy), TOP3: \(top3)"
        }
    }
    

    
    func saveParameters(_ params: [CGFloat]) {
        let expectedCount = {
            Self.layers.enumerated().map { (index, size) in
                if index < Self.layers.count - 1 {
                    return size + size * Self.layers[index + 1]
                } else {
                    return size
                }
            }.reduce(0, +) - Self.layers[0]
        }()
        
        guard params.count == expectedCount else { return }
        
        var currentIndex = 0
        
        // 1. Reconstruct inputWeights (layers[0] x layers[1])
        for i in 0..<Self.layers[0] {
            let nextIndex = currentIndex + Self.layers[1]
            inputWeights[i] = Array(params[currentIndex..<nextIndex])
            currentIndex = nextIndex
            
        }
        
        print("after inputs: \(currentIndex)")
                
        for (index, layer) in hiddenLayers.enumerated() {
            print("rep: \(currentIndex)")

            let biasesStartIndex = currentIndex
            let biasesEndIndex = currentIndex + layer.biases.count
            
            for i in biasesStartIndex..<biasesEndIndex {
                hiddenLayers[index].biases[i - biasesStartIndex] = params[i]
                currentIndex += 1
            }
            print("poop: \(currentIndex)")

            let weightsStartIndex = currentIndex
            let weightsEndIndex = currentIndex + layer.weights.count * layer.weights.first!.count
            
            for i in weightsStartIndex ..< weightsEndIndex {
                let relativeIndex = i - weightsStartIndex
                
                var rowNumDouble = (Double(relativeIndex) / Double(layer.weights[0].count))
                rowNumDouble.round(.down)
                let rowNum = Int(rowNumDouble)
                
                let columnNum = relativeIndex % layer.weights.first!.count
                
                hiddenLayers[index].weights[rowNum][columnNum] = params[i]
                
                currentIndex += 1
            }
        }
        
        print("after middle: \(currentIndex)")

        
        // 4. Reconstruct outputBiases (layers[2])
        let outputBiasesEnd = currentIndex + Self.layers.last!
        outputBiases = Array(params[currentIndex..<outputBiasesEnd])
        
        
        print(outputBiasesEnd, expectedCount)
    }
    
    func list() -> [CGFloat] {
        var list = [CGFloat]()
        
        for array in inputWeights {
            list.append(contentsOf: array)
        }
        
        for layer in hiddenLayers {
            list.append(contentsOf: layer.biases)
            list.append(contentsOf: layer.weights.flatMap(\.self))
        }
        
        list.append(contentsOf: outputBiases)
        
        return list
    }
    
    enum NetworkErrors: Error {
           case invalidInputs
       }
       
       
       //    func mapRange(_ x: CGFloat) -> CGFloat {
       //        return (2.0 / 9.0) * x - 1.0
       //    }
       
       enum RandomNumberType {
           case human
           case computer
           case logicError
       }
       
       static func cost(output: [CGFloat], answer: [CGFloat]) -> CGFloat {
           guard output.count == Self.layers.last else { return CGFloat.infinity }
           guard answer.count == Self.layers.last else { return CGFloat.infinity }
           
           var sum: CGFloat = 0
           
           for (output, answer) in zip(output, answer) {
               sum += pow(output - answer, 2)
           }
           
           
           return sum
       }
       
       func tanhDerivative(_ postActivation: CGFloat) -> CGFloat {
           return 1 - pow(postActivation, 2)
       }
       
       // MARK: - Backpropagation
       func backprop(iterations: Int, learnRate: CGFloat, batchSize: CGFloat) {
           for i in 0..<iterations {
               if i % 80 == 0 {
                   let v = evaluate(on: validationStarts)
                   print("Iteration \(i)/\(iterations) — \(v.printOut())")
                   scores.append(v.mse)
               }
               
               var outputBiasesPartials: [CGFloat] = Array(repeating: 0, count: Self.layers[2])
               var middleBiasesPartials: [CGFloat] = Array(repeating: 0, count: Self.layers[1])
               var middleWeightPartials: [[CGFloat]] = Array(repeating: Array(repeating: 0, count: Self.layers[2]), count: Self.layers[1])
               var inputWeightPartials: [[CGFloat]] = Array(repeating: Array(repeating: 0, count: Self.layers[1]), count: Self.layers[0])
               
               for _ in 0..<Int(batchSize) {
                   let seed = sampleSeed
                   let input = encode(startingAt: seed, noise: 0.03)
                   let target = makeTarget(startingAt: seed)
                   
                   let datum: ([CGFloat], [CGFloat]) = (input, target)
                   
                   
                   let (_, captured) = self.run(datum.0)
                   
                   for (index, outputNeuron) in captured.outputNeurons.enumerated() {
                       outputBiasesPartials[index] += 2 * (outputNeuron - datum.1[index]) * tanhDerivative(outputNeuron)
                   }
                   
                   for (index, middleNeuron) in captured.middleNeurons.enumerated() {
                       for (syndex, outputNeuron) in captured.outputNeurons.enumerated() {
                           middleWeightPartials[index][syndex] += 2 * (outputNeuron - datum.1[syndex]) * middleNeuron * tanhDerivative(outputNeuron) /* It's like how here, the partial doesn't depend on middleWeights[index][syndex]*/
                       }
                   }
                   
                   var middleDeltas: [CGFloat] = Array(repeating: 0, count: Self.layers[1])
                   
                   for (index, middleNeuron) in captured.middleNeurons.enumerated() {
                       var result: CGFloat = 0
                       for (syndex, outputNeuron) in captured.outputNeurons.enumerated() {
                           result += 2 * (outputNeuron - datum.1[syndex])
                           * tanhDerivative(outputNeuron)
                           * middleWeights[index][syndex]
                       }
                       let delta = result * tanhDerivative(middleNeuron)
                       middleDeltas[index] = delta
                       middleBiasesPartials[index] += delta
                   }
                   
                   for (index, inputNeuron) in captured.inputNeurons.enumerated() {
                       if inputNeuron == 0 { continue }
                       for syndex in 0..<Self.layers[1] {
                           inputWeightPartials[index][syndex] += middleDeltas[syndex] * inputNeuron
                       }
                   }
               }
               
               
               outputBiasesPartials = outputBiasesPartials.map { $0 * learnRate / batchSize }
               middleBiasesPartials = middleBiasesPartials.map { $0 * learnRate / batchSize }
               inputWeightPartials = inputWeightPartials.map { $0.map { $0 * learnRate / batchSize }}
               middleWeightPartials = middleWeightPartials.map { $0.map { $0 * learnRate / batchSize }}
               
               for (index, _) in outputBiases.enumerated() {
                   outputBiases[index] -= outputBiasesPartials[index]
               }
               
               for (index, _) in middleBiases.enumerated() {
                   middleBiases[index] -= middleBiasesPartials[index]
               }
               
               for (index, middleWeight) in middleWeights.enumerated() {
                   for (syndex, _) in middleWeight.enumerated() {
                       middleWeights[index][syndex] -= middleWeightPartials[index][syndex]
                   }
               }
               
               for (index, inputWeight) in inputWeights.enumerated() {
                   for (syndex, _) in inputWeight.enumerated() {
                       inputWeights[index][syndex] -= inputWeightPartials[index][syndex]
                   }
               }
           }
       }
       
       func prediction(from context: String) -> String {
           let inputs = generateOneHotVectors(for: context)
           let (outputs, _) = run(inputs)
           return softmax(outputs)
       }
       
       func generate(from context: String, characters: Int) -> String {
           var result = context
           for _ in 0..<characters {
               result.append(prediction(from: result))
           }
           return result
       }
       
       func argmax(_ outputs: [CGFloat]) -> String {
           var max: (value: CGFloat, index: Int) = (-CGFloat.infinity, 0)
           for (i, o) in outputs.enumerated() {
               if o > max.value { max.value = o; max.index = i }
           }
           return String(vocab[max.index])
       }
       
       func softmax(_ outputs: [CGFloat], temperature: CGFloat = 0.08) -> String {
           let scaled = outputs.map { $0 / temperature }
           let m = scaled.max()!
           let exps = scaled.map { exp(($0 - m)) }
           let sum = exps.reduce(0, +)
           
           var r = CGFloat.random(in: 0..<sum)
           for (k, e) in exps.enumerated() {
               r -= e
               if r <= 0 { return String(vocab[k]) }
           }
           return String(vocab[exps.count - 1])
       }
       
       
       func run(_ inputs: [CGFloat]) -> ([CGFloat], CapturedNeurons) {
           guard inputs.count == Self.layers[0] else {
               print("FATAL ERROR: Input count mismatch")
               fatalError()
           }
           
           let mapped = inputs // Can apply mapping
           var neurons: CapturedNeurons = .init()
           
           neurons.inputNeurons = mapped
           
           var middleValues: [CGFloat] = Array(repeating: 0.0, count: Self.layers[1])
           for middleNeuron in 0..<Self.layers[1] {
               var sum: CGFloat = 0
               
               for (index, input) in mapped.enumerated() {
                   if input == 0 { continue }
                   
                   let scaled = input * inputWeights[index][middleNeuron]
                   
                   sum += scaled
               }
               
               sum += middleBiases[middleNeuron]
               
               
               middleValues[middleNeuron] = tanh(sum)
           }
           
           neurons.middleNeurons = middleValues
           
           var outputValues: [CGFloat] = Array(repeating: 0.0, count: Self.layers[2])
           for outputNeuron in 0..<Self.layers[2] {
               var sum: CGFloat = 0
               
               for (index, input) in middleValues.enumerated() {
                   let scaled = input * middleWeights[index][outputNeuron]
                   
                   sum += scaled
               }
               
               sum += outputBiases[outputNeuron]
               
               
               outputValues[outputNeuron] = tanh(sum)
           }
           
           neurons.outputNeurons = outputValues
           
           return (outputValues, neurons)
       }
       func evaluate(on starts: [Int]) -> Metrics {
           var squaredError: CGFloat = 0
           var correct = 0
           var inTop3 = 0
           
           for start in starts {
               let input = encode(startingAt: start)
               let target = makeTarget(startingAt: start)
               let answer = sampleData[start + textContextSize]
               let output = self.run(input).0
               
               for k in 0..<27 {
                   let e = output[k] - target[k]
                   squaredError += e * e
               }
               
               // argmax
               var best = 0
               for k in 1..<27 where output[k] > output[best] { best = k }
               if best == answer { correct += 1 }
               
               // is the true answer among the three highest outputs?
               var betterCount = 0
               for k in 0..<27 where output[k] > output[answer] { betterCount += 1 }
               if betterCount < 3 { inTop3 += 1 }
           }
           
           let n = CGFloat(starts.count)
           
           return Metrics(
               mse: squaredError / (n * 27),
               accuracy: CGFloat(correct) / n,
               top3: CGFloat(inTop3) / n
           )
       }
    
}


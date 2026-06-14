import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;

public class Driver {

    public static void main(String[] args) throws Exception {
        KaggleConfig config = new KaggleConfig();

        System.out.println("--- Actuarial Claims Severity Engine ---");

        // ---> NEW: Ensure reproducibility before anything else happens <---
        DataSplitter.generateSplitsIfMissing(config);

        // 1. SCAN THE UNIVERSE: Pass the RAW Kaggle files to build the 1190 dictionary
        KaggleParser parser = new KaggleParser(config.rawSourcePath, config.rawKaggleTestPath);
        // ... rest of main method ...

        // ---> FIX: Explicitly trigger the one-time metadata scan here <---
        System.out.println("Initiating one-time metadata scan...");
        parser.scanMetadata();
        config.nodes[0] = parser.totalInputFeatures;
        // 2. Inject the parser into the file streaming engines
        StreamingIterator[] iterators = loadData(config, parser);
        // NeuralNetwork model = createModel(config);
        NeuralNetwork model = loadModel(config);
        // 3. Execute the pipeline
        trainModel(model, iterators[0], iterators[1], config);
        evaluatePredictions(model, iterators[2], config);

        // 4. Generate the final file for Kaggle submission
        generateKaggleSubmission(model, parser, config.rawKaggleTestPath, config.rawKaggleSubmissionPath);
    }

    // --- Extracted Methods ---

    private static StreamingIterator[] loadData(KaggleConfig config, KaggleParser parser) throws Exception {
        System.out.println("Loading data streams...");
        // Parameters: batchSize, filePath, hasHeader, parser
        StreamingIterator trainer = new StreamingIterator(config.batchSize, config.trainPath, true, parser);
        StreamingIterator validator = new StreamingIterator(config.batchSize, config.validationPath, true, parser);
        StreamingIterator tester = new StreamingIterator(config.batchSize, config.testPath, true, parser);
        return new StreamingIterator[] { trainer, validator, tester };
    }

    private static NeuralNetwork loadModel(KaggleConfig config) throws Exception {
        System.out.println("Loading neural network architecture...");
        NeuralNetwork model = NeuralNetwork.load(config.loadModelPath);
        model.printStructure();
        return model;
    }

    private static NeuralNetwork createModel(KaggleConfig config) throws Exception {
        System.out.println("Creating neural network architecture...");
        NeuralNetwork model = NeuralNetwork.createModel(config.nodes, config.activations, config.costFunction);
        model.printStructure();
        return model;
    }

    private static void trainModel(NeuralNetwork model, StreamingIterator trainer, StreamingIterator validator,
            KaggleConfig config) throws Exception {
        System.out.println("\nInitializing training pipeline...");
        System.out.println("Batch Size: " + config.batchSize + " | Epochs: " + config.epochs);

        // 1. Calculate the baseline error before any training happens
        double bestValCost = calculateActualDollarMAE(model, validator);

        // 2. Explicitly print the baseline so it shows up even if epochs = 0
        System.out.printf("Baseline Validation Mean Absolute Error (Epoch 0): $%,10.2f\n\n", bestValCost);

        double currentLearningRate = config.learningRate;
        long totalStartTime = System.currentTimeMillis();

        // 3. The loop (will be skipped entirely if config.epochs == 0)
        for (int epoch = 1; epoch <= config.epochs; epoch++) {
            long epochStartTime = System.currentTimeMillis();

            // Capture the statistics returned from the training run
            int[] trainingStats = model.train(trainer, currentLearningRate);
            int batchesTrained = trainingStats[0];
            int dataPointsTrained = trainingStats[1];

            double averageDollarError = calculateActualDollarMAE(model, validator);

            if (averageDollarError < bestValCost) {
                System.out.println(">>> New best model found! Saving weights to " + config.saveModelPath);
                bestValCost = averageDollarError;
                model.save(config.saveModelPath);
            }

            double epochTimeSec = (System.currentTimeMillis() - epochStartTime) / 1000.0;

            // Added Batches and Data points (with commas for readability) to the output
            // string
            // Replace your current System.out.printf with this concise version:
            System.out.printf("[Epoch %d | %5.2fs] LR: %.5f | Rows: %,d | Val MAE: $%,.2f\n",
                    epoch, epochTimeSec, currentLearningRate, dataPointsTrained, averageDollarError);

            currentLearningRate *= config.decayRate;
        }

        double totalTimeMin = (System.currentTimeMillis() - totalStartTime) / 60000.0;
        System.out.printf("\nPipeline Complete in %.2f minutes!\n", totalTimeMin);
    }

    private static void evaluatePredictions(NeuralNetwork model, StreamingIterator tester, KaggleConfig config)
            throws Exception {
        System.out.println("\n--- Out-of-Sample Predictions ---");
        DataPair[] dataPairs = tester.get(0, config.displayCount);

        for (DataPair pair : dataPairs) {
            // The parser already normalized pair.input, so we pass it straight into
            // compute()
            double expectedDollar = Math.exp(pair.label.get(0, 0));
            double actualDollar = Math.exp(model.compute(pair.input).get(0, 0));

            System.out.printf("Expected: $%,10.2f | Predicted: $%,10.2f\n", expectedDollar, actualDollar);
            Thread.sleep(100);
        }
    }

    private static double calculateActualDollarMAE(NeuralNetwork model, StreamingIterator validator) throws Exception {
        double totalDollarErrorSum = 0;
        int totalRecordsEvaluated = 0;

        // 1. Stream through the validation set
        while (validator.hasNextBatch()) {
            for (DataPair pair : validator.nextBatch()) {

                // 2. Compute the prediction
                double predictedLog = model.compute(pair.input).get(0, 0);
                double expectedLog = pair.label.get(0, 0);

                // 3. Revert back to real Actuarial Dollars using Math.exp()
                double expectedDollar = Math.exp(expectedLog);
                double predictedDollar = Math.exp(predictedLog);

                // 4. Accumulate the absolute dollar difference
                totalDollarErrorSum += Math.abs(expectedDollar - predictedDollar);
                totalRecordsEvaluated++;
            }
        }

        // 5. Calculate the average Dollar MAE
        double averageDollarError = totalDollarErrorSum / totalRecordsEvaluated;

        // 6. CRITICAL: Reset the iterator so it's ready for the next training epoch
        validator.reset();

        return averageDollarError;
    }

    private static void generateKaggleSubmission(NeuralNetwork model, KaggleParser parser,
            String kaggleTestPath, String outputPath) throws Exception {
        System.out.println("\nGenerating Kaggle Submission File...");

        BufferedReader reader = new BufferedReader(new FileReader(kaggleTestPath));
        PrintWriter writer = new PrintWriter(outputPath);

        // Write the Kaggle required header
        writer.println("id,loss");
        reader.readLine(); // Skip the input file's header

        String line;
        int count = 0;
        while ((line = reader.readLine()) != null) {
            // 1. Get the ID (always the first column)
            String id = line.split(",")[0];

            // 2. Parse and normalize the unlabeled data using our custom logic
            Matrix normalizedInput = parser.parseUnlabeledLine(line);

            // 3. Compute the log-loss and revert to Actuarial Dollars
            double logPrediction = model.compute(normalizedInput).get(0, 0);
            double dollarPrediction = Math.exp(logPrediction);

            // 4. Write directly to the submission file
            writer.printf("%s,%.2f\n", id, dollarPrediction);
            count++;
        }

        reader.close();
        writer.close();
        System.out.println("Successfully wrote " + count + " predictions to " + outputPath);
    }
}
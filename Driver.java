public class Driver {
    static int batchSize = 128;
    static double learningRate = .001;
    static double decayRate = 0.995; // This shrinks the learning rate by 15% each epoch
    static int epochs = 50;
    static int timeToDisplay = 1;
    static int imageCount = 50;
    static double scale = 10.;
    static NeuralNetwork mnist;
    static DataIterator trainer, validator, tester;
    static String saveLocation = "./serials/newbest.ser";

    public static void main(String[] args) throws Exception {
        trainer = new DataIterator(batchSize, "./trainHalfDataPairs.ser");
        validator = new DataIterator(batchSize, "./validateDataPairs.ser");
        tester = new DataIterator(batchSize, "./testDataPairs.ser");
        // mnist = new NeuralNetwork(1190, "leakyRelu");
        // mnist.add(512, "leakyRelu");
        // mnist.add(128, "leakyRelu");
        // mnist.add(32, "identity");
        // mnist.add(1);
        // mnist.compile("meanAbsoluteError");
        mnist = NeuralNetwork.load("./serials/showingoff.ser");
        mnist.printStructure();
        // Initialize the DataIterators
        double bestValCost, averageCost;
        bestValCost = averageCost = mnist.validate(validator);
        mnist.save(saveLocation);
        System.out.println("Initial Statistics:");
        System.out.println("Average Cost: " + averageCost);
        System.out.println("Batch Size: " + batchSize);
        System.out.println("Number of Batches: " + trainer.numBatches);
        System.out.println("Number of Epochs: " + epochs);
        System.out.println("Initializing training...\r\n");
        long totalStartTime = System.currentTimeMillis();
        for (int i = 1; i < epochs + 1; i++) {
            long epochStartTime = System.currentTimeMillis();
            mnist.train(trainer, learningRate);
            averageCost = mnist.validate(validator);
            if (averageCost < bestValCost) {
                System.out.println("New best model found! Saving weights...");
                bestValCost = averageCost;
                mnist.save(saveLocation);
            }
            long epochEndTime = System.currentTimeMillis();
            double epochTimeSec = (epochEndTime - epochStartTime) / 1000.0;
            System.out.println("Learning Rate: " + learningRate);
            System.out.println("Average Cost: " + averageCost);
            System.out.println("Epoch " + i + " complete in " + epochTimeSec + " seconds\r\n");
            learningRate *= decayRate;
        }
        long totalEndTime = System.currentTimeMillis();
        double totalTimeMin = (totalEndTime - totalStartTime) / 60000.0;
        System.out.println(String.format("Training Complete in %.2f minutes!\r\n", totalTimeMin));
        // Showcase of images and predictions here

        DataPair[] dataPairs = tester.get(0, imageCount);
        for (int i = 0; i < dataPairs.length; i++) {
            Matrix normalizedInput = dataPairs[i].input.clone();
            normalizedInput.minus(tester.mean).divide(tester.standardDeviation);
            double expectedIndex = Math.exp(dataPairs[i].label.get(0, 0));
            double actualIndex = Math.exp(mnist.compute(normalizedInput).get(0, 0));
            System.out.printf("Expected: $%.2f" + ", Output: $%.2f" + "\r\n", expectedIndex, actualIndex);
            Thread.sleep(timeToDisplay);

        }
        return;
    }
}

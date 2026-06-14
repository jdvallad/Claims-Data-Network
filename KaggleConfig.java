public class KaggleConfig extends CoreConfig {
    public int displayCount = 50;
    public String rawKaggleTestPath = "./csv/test.csv";
    public String rawKaggleSubmissionPath = "./csv/submission.csv";

    public KaggleConfig() {
        // Defaults are set above
        // Hyperparameters
        this.batchSize = 128;
        this.learningRate = 0.001;
        this.decayRate = 0.995;
        this.epochs = 20;

        // --- SPLIT SETTINGS ---
        this.randomSeed = 42L; // The Answer to the Ultimate Question (ensures reproducibility)
        this.trainSplit = 0.80; // 80% Train
        this.validationSplit = 0.10; // 10% Validate
        // Test split is implicitly the remaining 10%
        // Execution Settings

        // File Paths
        this.rawSourcePath = "./csv/train.csv";
        this.trainPath = "./csv/train_split.csv";
        this.validationPath = "./csv/validate_split.csv";
        this.testPath = "./csv/test_split.csv";
        this.loadModelPath = "./models/v3.ser";
        this.saveModelPath = "./models/newBestHopefully.ser";
        this.nodes = new int[] { 0, 512, 128, 1 }; // input layer will be updated after hot-encoding
        this.activations = new String[] { "leakyRelu", "leakyRelu", "identity" };
        this.costFunction = "meanSquaredError";

    }
}
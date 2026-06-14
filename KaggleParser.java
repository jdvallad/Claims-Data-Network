import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class KaggleParser implements DataParser<String> {

    private String trainPath;
    private String testPath;

    public double mean, standardDeviation;
    private Map<Integer, Map<String, Integer>> categoryMaps;
    private boolean[] isNumeric;
    public int totalInputFeatures;

    public KaggleParser(String trainPath, String testPath) {
        this.trainPath = trainPath;
        this.testPath = testPath;
        this.categoryMaps = new HashMap<>();
    }

    @Override
    public void scanMetadata() throws Exception {
        // Temporary storage to collect unique strings before sorting
        Map<Integer, Set<String>> uniqueCategories = new HashMap<>();

        // --- PART A: Scan the Training Data ---
        BufferedReader trainReader = new BufferedReader(new FileReader(trainPath));
        trainReader.readLine(); // Skip header

        String firstLine = trainReader.readLine();
        if (firstLine == null) {
            trainReader.close();
            return;
        }
        String[] firstValues = firstLine.split(",");
        int trainNumCols = firstValues.length;
        this.isNumeric = new boolean[trainNumCols];

        for (int i = 1; i < trainNumCols - 1; i++) {
            if (firstValues[i].matches("-?\\d+(\\.\\d+)?")) {
                this.isNumeric[i] = true;
            } else {
                this.isNumeric[i] = false;
                uniqueCategories.put(i, new HashSet<>());
            }
        }

        long numericValueCount = 0;
        double sum = 0;
        double sqSum = 0;

        String currentLine = firstLine;
        while (currentLine != null) {
            String[] values = currentLine.split(",");
            for (int i = 1; i < trainNumCols - 1; i++) {
                if (isNumeric[i]) {
                    double val = Double.parseDouble(values[i]);
                    sum += val;
                    sqSum += val * val;
                    numericValueCount++;
                } else {
                    // Just collect the unique string, don't assign an index yet!
                    uniqueCategories.get(i).add(values[i]);
                }
            }
            currentLine = trainReader.readLine();
        }
        trainReader.close();

        this.mean = sum / numericValueCount;
        double variance = (sqSum / numericValueCount) - (this.mean * this.mean);
        this.standardDeviation = Math.sqrt(variance);

        // --- PART B: Scan the Kaggle Test Data (Expand Categories) ---
        if (testPath != null && !testPath.isEmpty()) {
            System.out.println("Expanding one-hot dictionaries using " + testPath + "...");
            BufferedReader testReader = new BufferedReader(new FileReader(testPath));
            testReader.readLine(); // Skip header

            String testLine;
            while ((testLine = testReader.readLine()) != null) {
                String[] values = testLine.split(",");
                for (int i = 1; i < values.length; i++) {
                    if (!isNumeric[i]) {
                        uniqueCategories.get(i).add(values[i]);
                    }
                }
            }
            testReader.close();
        }

        // --- PART C: Sort and Finalize the Dictionaries ---
        this.totalInputFeatures = 0;
        for (int i = 1; i < trainNumCols - 1; i++) {
            if (isNumeric[i]) {
                this.totalInputFeatures += 1;
            } else {
                // 1. Convert the Set to a List
                List<String> sortedList = new ArrayList<>(uniqueCategories.get(i));

                // 2. Sort by Length, then Alphabetically (e.g., A, B ... Z, AA, AB)
                Collections.sort(sortedList, (s1, s2) -> {
                    if (s1.length() != s2.length()) {
                        return Integer.compare(s1.length(), s2.length());
                    }
                    return s1.compareTo(s2);
                });

                // 3. Map them to their permanent sorted integer index
                Map<String, Integer> finalMap = new HashMap<>();
                for (int j = 0; j < sortedList.size(); j++) {
                    finalMap.put(sortedList.get(j), j);
                }

                this.categoryMaps.put(i, finalMap);
                this.totalInputFeatures += finalMap.size();
            }
        }
    }

    @Override
    public DataPair parse(String line) throws Exception {
        String[] values = line.split(",");

        double[] input2D = new double[this.totalInputFeatures];
        int featureIndex = 0;

        for (int i = 1; i < values.length - 1; i++) {
            if (isNumeric[i]) {
                // NORMALIZATION FIX: Apply mean/std ONLY to the continuous values
                double rawValue = Double.parseDouble(values[i]);
                input2D[featureIndex++] = (rawValue - this.mean) / this.standardDeviation;
            } else {
                int categoryIndex = categoryMaps.get(i).get(values[i]);
                int totalCategories = categoryMaps.get(i).size();
                for (int j = 0; j < totalCategories; j++) {
                    // Leave the categorical flags as pure 1.0 or 0.0
                    input2D[featureIndex++] = (j == categoryIndex) ? 1.0 : 0.0;
                }
            }
        }

        // Create the Matrix (Do NOT call .minus() or .divide() here anymore!)
        Matrix inputMatrix = Matrix.create(input2D);

        double loss = Double.parseDouble(values[values.length - 1]);
        double logLoss = Math.log(loss);
        Matrix labelMatrix = Matrix.create(new double[] { logLoss });

        return new DataPair(inputMatrix, labelMatrix);
    }

    // Custom method for the final submission output
    public Matrix parseUnlabeledLine(String line) throws Exception {
        String[] values = line.split(",");

        double[] input2D = new double[this.totalInputFeatures];
        int featureIndex = 0;

        for (int i = 1; i < values.length; i++) {
            if (isNumeric[i]) {
                // NORMALIZATION FIX: Apply mean/std ONLY to the continuous values
                double rawValue = Double.parseDouble(values[i]);
                input2D[featureIndex++] = (rawValue - this.mean) / this.standardDeviation;
            } else {
                int categoryIndex = categoryMaps.get(i).get(values[i]);
                int totalCategories = categoryMaps.get(i).size();
                for (int j = 0; j < totalCategories; j++) {
                    // Leave the categorical flags as pure 1.0 or 0.0
                    input2D[featureIndex++] = (j == categoryIndex) ? 1.0 : 0.0;
                }
            }
        }

        // Create the Matrix (Do NOT call .minus() or .divide() here anymore!)
        Matrix inputMatrix = Matrix.create(input2D);

        return inputMatrix;
    }
}
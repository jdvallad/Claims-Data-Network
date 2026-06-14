import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class script {

    public static void main(String[] args) throws Exception {
        System.out.println("Starting serialization...");
        String filePath = "./";

        List<Map<String, Object>> data = new ArrayList<>();
        HashMap<String, HashMap<String, Integer>> dictionary = new HashMap<>();
        int continousCount = 14;
        int vectorLength = getDictionary("schema mapping.csv", ",", dictionary);

        String line;
        String[] columnNames = new String[0];
        int dataLineCount = 0;
        try (BufferedReader br = new BufferedReader(new FileReader("train_half.csv"))) {
            while ((line = br.readLine()) != null) {
                if (line.trim().equals("")) {
                    continue;
                }
                dataLineCount++;
                if (dataLineCount == 1) {
                    columnNames = line.split("\\s{0,}" + "," + "\\s{0,}");
                    continue;
                }
                String[] lineParts = line.split("\\s{0,}" + "," + "\\s{0,}");
                DataPair temp;
                double[] input = new double[vectorLength + continousCount];
                double[] output = new double[1];
                int lineIndex = 1;
                int inputIndex = vectorLength;
                while (lineIndex < dictionary.size() + 1) {
                    int index = dictionary.get(columnNames[lineIndex]).get(lineParts[lineIndex]);
                    input[index] = 1;
                    lineIndex++;
                }
                while (lineIndex < dictionary.size() + continousCount + 1) {
                    input[inputIndex] = Double.parseDouble(lineParts[lineIndex]);
                    lineIndex++;
                    inputIndex++;
                }
                output[0] = Math.log(Double.parseDouble(lineParts[dictionary.size() + continousCount + 1]));
                temp = new DataPair(Matrix.create(input), Matrix.create(output));
                data.add(temp.save());
            }
        }
        // Trap these Exceptions
        catch (FileNotFoundException ex) {
            System.err.println(ex.getMessage());
        } catch (IOException ex) {
            System.err.println(ex.getMessage());
        }

        Collections.shuffle(data);
        FileOutputStream fileOut = new FileOutputStream(filePath + "trainHalfDataPairs.ser");
        ObjectOutputStream out = new ObjectOutputStream(fileOut);
        out.writeObject(data);
        out.close();
        fileOut.close();
        System.out.println("Created file at " + filePath + "trainHalfDataPairs.ser");
        System.exit(0);
    }

    // Pass columnNames into this method
    public static int createOrderedDictionary(HashMap<String, HashMap<String, Integer>> freqMap,
            HashMap<String, HashMap<String, Integer>> dictionary,
            String[] columnNames) {
        int globalIndex = 0;

        // FORCING THE ORDER: Loop from cat1 (index 1) to cat116 (index 116)
        for (int i = 0; i < 116; i++) {
            String colName = columnNames[i];

            // Safety check in case a column somehow has no data
            if (!freqMap.containsKey(colName))
                continue;

            List<String> categories = new ArrayList<>(freqMap.get(colName).keySet());

            // Simple sort: length first, then alphabetical
            Collections.sort(categories, new Comparator<String>() {
                @Override
                public int compare(String a, String b) {
                    if (a.length() != b.length()) {
                        return Integer.compare(a.length(), b.length());
                    }
                    return a.compareTo(b);
                }
            });

            HashMap<String, Integer> colDict = new HashMap<>();
            for (String cat : categories) {
                colDict.put(cat, globalIndex);
                globalIndex++;
            }
            dictionary.put(colName, colDict);
        }
        return globalIndex;
    }

    public static int getDictionary(String csvFilePath, String csvDelimiter,
            HashMap<String, HashMap<String, Integer>> dictionary) {
        HashMap<String, HashMap<String, Integer>> map = new HashMap<>();
        String line;
        String[] columnNames = new String[0];
        int dataLineCount = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(csvFilePath))) {
            while ((line = br.readLine()) != null) {
                if (line.trim().equals("")) {
                    continue;
                }
                dataLineCount++;
                if (dataLineCount == 1) {
                    columnNames = line.split("\\s{0,}" + csvDelimiter + "\\s{0,}");
                    for (int i = 0; i < columnNames.length; i++) {
                        map.put(columnNames[i], new HashMap<>());
                    }
                    continue;
                }
                String[] lineParts = line.split("\\s{0,}" + csvDelimiter + "\\s{0,}");
                for (int i = 0; i < columnNames.length; i++) {
                    HashMap<String, Integer> temp = map.get(columnNames[i]);
                    if (!temp.containsKey(lineParts[i])) {
                        temp.put(lineParts[i], 1);
                    } else {
                        temp.put(lineParts[i], temp.get(lineParts[i]) + 1);
                    }
                }
            }
            return createOrderedDictionary(map, dictionary, columnNames);
        }
        // Trap these Exceptions
        catch (FileNotFoundException ex) {
            System.err.println(ex.getMessage());
        } catch (IOException ex) {
            System.err.println(ex.getMessage());
        }
        return -1;
    }

    // public static int createOrderedDictionary(HashMap<String, HashMap<String,
    // Integer>> freqMap,
    // HashMap<String, HashMap<String, Integer>> dictionary) {
    // int globalIndex = 0;

    // for (String colName : freqMap.keySet()) {
    // List<String> categories = new ArrayList<>(freqMap.get(colName).keySet());

    // // Simple sort: length first, then alphabetical
    // Collections.sort(categories, new Comparator<String>() {
    // @Override
    // public int compare(String a, String b) {
    // if (a.length() != b.length()) {
    // return Integer.compare(a.length(), b.length());
    // }
    // return a.compareTo(b);
    // }
    // });

    // HashMap<String, Integer> colDict = new HashMap<>();
    // for (String cat : categories) {
    // colDict.put(cat, globalIndex);
    // globalIndex++;
    // }
    // dictionary.put(colName, colDict);
    // }
    // return globalIndex; // This is the starting index for your continuous
    // variables
    // }
}

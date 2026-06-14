import java.io.*;
import java.util.*;

public class PeekFile {
    public static void main(String[] args) throws Exception {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("dataPairs.ser"))) {
            List<Map<String, Object>> data = (List<Map<String, Object>>) ois.readObject();
            System.out.println("Successfully read " + data.size() + " pairs.");
            DataPair.load(data.get(0)).input.printDimensions();
            System.out.println();
            DataPair.load(data.get(0)).label.printDimensions();
        }
    }
}